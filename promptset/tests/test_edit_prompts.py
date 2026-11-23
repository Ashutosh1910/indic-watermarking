"""Tests for promptset/scripts/build_edit_prompts.py (edit-level prompt builder).

Run from the repo root:  python3 -m pytest promptset/tests/test_edit_prompts.py -q
(or python3 -m unittest promptset.tests.test_edit_prompts). No tokenizer is needed: token counts are
estimated from tokens_per_word, and the exact-tokenizer path is exercised with a fake counter.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "promptset" / "scripts"))
sys.path.insert(0, str(REPO))

import build_edit_prompts as bep  # noqa: E402

HAVE_CORPORA = (REPO / "data" / "Samalochan").is_dir() and (REPO / "data" / "Hindi-Aesthetics-Corpus").is_dir()
REQUIRED = ("id", "pair_id", "language", "level", "genre", "source", "source_id", "license", "split", "experiment",
            "input_cutoff_kind", "input_cutoff", "input_words", "input_tokens", "human_passage", "k",
            "target_words", "max_new_tokens", "cutoffs", "template_id", "messages", "chat_template_kwargs",
            "generation", "watermark")
N_DOCS = 6


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ceil64(x: float) -> int:
    return int(math.ceil(x / 64) * 64)


def words_of(s: str) -> int:
    return len(s.split())


def ends_in_words(text: str, pattern) -> list[int]:
    starts = [m.start() for m in re.finditer(r"\S+", text)]
    return [bisect.bisect_left(starts, e) for e in bep.sentence_ends(text, pattern)]


def run_build(out: Path, *extra: str) -> None:
    bep.main(["build", "--tokenizer", "none", "--out", str(out), *extra])


@unittest.skipUnless(HAVE_CORPORA, "Hindi corpora not under data/")
class TestBuildFromCorpora(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name) / "a"
        run_build(cls.out, "--corpus", "samalochan", "--corpus", "hindi-aesthetics", "--n-docs", str(N_DOCS))
        cls.rows = read(cls.out / "edit_prompts.jsonl")
        cls.passages = read(cls.out / "human_passages.jsonl")
        cls.cfg = bep.load_config()
        import corpora

        cls.docs = {d.id: d for name in ("samalochan", "hindi-aesthetics") for d in corpora.get(name)}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_counts(self):
        per_source = Counter(p["pair_id"] for p in self.passages)
        sources = Counter(p.split("/")[0] for p in per_source)
        self.assertEqual(sources, {"samalochan": N_DOCS, "hindi-aesthetics": N_DOCS})
        levels = Counter(bep.level_label(r) for r in self.rows)
        for lv in ("L3_paraphrase", "L4_rewrite", "C_continue_k75", "C_continue_k50", "C_continue_k25",
                   "OX_extract_outline"):
            self.assertEqual(levels[lv], len(self.passages), lv)

    def test_schema(self):
        exps = set(self.cfg["experiments"])
        for r in self.rows:
            for k in REQUIRED:
                self.assertIn(k, r, f"{r['id']} lacks {k}")
            self.assertIn(r["experiment"], exps)
            self.assertEqual(r["language"], "hi")
            self.assertEqual(r["source_id"], r["pair_id"])
            self.assertEqual(r["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual([m["role"] for m in r["messages"]], ["system", "user"])
            self.assertEqual(r["messages"][0]["content"], self.cfg["kit_data"]["system"]["hi"])
            self.assertIsInstance(r["generation"]["seed"], int)
            self.assertEqual(set(r["cutoffs"]), {"words", "tokens"})
            self.assertRegex(r["template_id"], r"^[A-Za-z0-9_]+/\d+$")
            self.assertEqual(r["template_id"].split("/")[0], r["level"])
            self.assertTrue(r["token_counts_estimated"])
            self.assertEqual(r["genre"], "unknown")
            self.assertTrue(r["license"])
            self.assertEqual(r["max_new_tokens"] % 64, 0)
            self.assertEqual(r["watermark"], r["level"] != "OX_extract_outline")
            self.assertNotRegex(r["messages"][1]["content"], r"\{(text|head|points|title|prompt|n)\}")
            if r["level"] != "C_continue":
                self.assertIsNone(r["k"])

    def test_max_new_tokens(self):
        for r in self.rows:
            lv, t = r["level"], r["input_tokens"]
            if lv.startswith("L"):
                self.assertEqual(r["max_new_tokens"], ceil64(1.5 * t))
                self.assertEqual(r["target_words"], r["input_words"])
                self.assertIn(r["human_passage"], r["messages"][1]["content"])
            elif lv == "C_continue":
                self.assertEqual(r["max_new_tokens"], ceil64(1.3 * (t - r["head_tokens"])))
            elif lv == "G_title":
                self.assertEqual(r["max_new_tokens"], ceil64(1.3 * t))
                self.assertIn(r["title"], r["messages"][1]["content"])
                self.assertIn(str(r["target_words"]), r["messages"][1]["content"])
                self.assertLessEqual(abs(r["target_words"] - r["input_words"]), 5)
            elif lv == "OX_extract_outline":
                self.assertEqual(r["max_new_tokens"], 1024)
                self.assertEqual(r["cutoffs"], {"words": [], "tokens": []})
                self.assertIn("passage_cutoffs", r)

    def test_passages_are_doc_prefixes_cut_at_sentence_ends(self):
        L = self.cfg["lengths"]
        counter = bep.TokenCounter(None, L["tokens_per_word"])
        for p in self.passages:
            doc = self.docs[p["pair_id"]]
            pr = doc.meta["punctuation_removed"]
            pattern = bep.sentence_pattern(pr)
            start = p["meta"]["passage_char_start"]
            end = start + len(p["text"])
            self.assertEqual(doc.text[start:end], p["text"], "passage must be a verbatim slice of the doc")
            body = doc.text[start:]
            self.assertIn(len(p["text"]), bep.sentence_ends(body, pattern), f"{p['id']} does not end at a sentence end")
            kind, n = p["input_cutoff_kind"], p["input_cutoff"]
            units = p["n_words"] if kind == "words" else p["n_tokens"]
            self.assertEqual(p["n_words"], words_of(p["text"]))
            self.assertEqual(p["n_tokens"], counter(p["text"], "hi"))
            self.assertLessEqual(units, n)
            self.assertGreaterEqual(units, 0.95 * n, f"{p['id']}: 95% rule")
            # largest cutoff the doc supports: no bigger cutoff of the same kind would be accepted
            bigger = [m for m in L["cutoffs"][kind] if m > n]
            for m in bigger:
                self.assertFalse(bep.cut_passage(body, "hi", kind, m, counter, L["min_frac"], pattern).accepted)
            # served cutoffs: exactly those at which the passage itself trims to an accepted prefix
            for k2 in ("words", "tokens"):
                for m in L["cutoffs"][k2]:
                    acc = bep.cut_passage(p["text"], "hi", k2, m, counter, L["min_frac"], pattern).accepted
                    self.assertEqual(acc, m in p["cutoffs"][k2])
                    self.assertTrue(m <= (p["n_words"] if k2 == "words" else p["n_tokens"]) / 0.95 or not acc)
            self.assertIn(n, p["cutoffs"][kind])
            self.assertEqual(p["sentence_end_rule"], "punct+linebreak" if pr else "punct")

    def test_rows_match_their_passage(self):
        by_id = {p["id"]: p for p in self.passages}
        for r in self.rows:
            p = by_id[r["passage_id"]]
            self.assertEqual(r["human_passage"], p["text"])
            self.assertEqual(r["input_words"], p["n_words"])
            if r["level"] != "OX_extract_outline":
                self.assertEqual(r["cutoffs"], p["cutoffs"])

    def test_continuation_heads(self):
        for r in (r for r in self.rows if r["level"] == "C_continue"):
            passage, head, k = r["human_passage"], r["head"], r["k"]
            self.assertTrue(passage.startswith(head))
            pattern = bep.sentence_pattern(r["meta"]["punctuation_removed"])
            total = words_of(passage)
            target = total * k / 100
            hw = words_of(head)
            self.assertEqual(hw, r["head_words"])
            inner = [w for w in ends_in_words(passage, pattern) if 0 < w < total]
            self.assertIn(hw, inner, "head ends at a sentence end")
            # nearest sentence end to k% -> within one sentence of the target: no end strictly in between
            self.assertTrue(all(abs(hw - target) <= abs(w - target) for w in inner))
            lo, hi = sorted((hw, target))
            self.assertFalse([w for w in inner if lo < w < hi], "another sentence end lies between head and k%")
            self.assertEqual(r["target_words"], bep.round_to(total - hw, 10))
            self.assertIn(head, r["messages"][1]["content"])
            self.assertIn(f"{r['target_words']} ", r["messages"][1]["content"])

    def test_punctuation_removed_policy(self):
        ha = [r for r in self.rows if r["source"] == "hindi-aesthetics"]
        self.assertTrue(ha)
        self.assertFalse([r for r in ha if r["level"] in ("L1_proofread", "L2_polish")])
        self.assertTrue(all(r["meta"]["punctuation_removed"] for r in ha))
        sam = [r for r in self.rows if r["source"] == "samalochan"]
        self.assertTrue([r for r in sam if r["level"] == "L1_proofread"])
        self.assertFalse(any(r["meta"]["punctuation_removed"] for r in sam))

    def test_titles(self):
        for r in (r for r in self.rows if r["level"] == "G_title"):
            self.assertIn(r["template_id"], ("G_title/0", "G_title/2"))
            ok, why = bep.usable_title(r["title"], "hi", self.cfg["title_check"])
            self.assertTrue(ok, why)
            self.assertFalse(re.search(r"[A-Za-z]", r["title"]))

    def test_ids_unique_and_stable(self):
        ids = [r["id"] for r in self.rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len({p["id"] for p in self.passages}), len(self.passages))
        out2 = Path(self.tmp.name) / "b"
        run_build(out2, "--corpus", "hindi-aesthetics", "--corpus", "samalochan", "--n-docs", str(N_DOCS))
        for name in ("edit_prompts.jsonl", "human_passages.jsonl"):
            self.assertEqual((self.out / name).read_bytes(), (out2 / name).read_bytes(), name)
        # a bigger sample keeps every earlier doc (stable sample) with identical rows
        out3 = Path(self.tmp.name) / "c"
        run_build(out3, "--corpus", "samalochan", "--n-docs", str(N_DOCS + 4))
        big = {r["id"]: r for r in read(out3 / "edit_prompts.jsonl")}
        for r in (r for r in self.rows if r["source"] == "samalochan"):
            self.assertEqual(big.get(r["id"]), r)

    def test_ox_build_outline_round_trip(self):
        ox = [r for r in self.rows if r["level"] == "OX_extract_outline"]
        self.assertTrue(ox)
        fake = []
        for i, r in enumerate(ox):
            if i == 0:
                text, fr = "यहाँ रूपरेखा है:\n\n- पहला बिंदु\n- दूसरा बिंदु", "eos"          # 2 bullets -> dropped
            elif i == 1:
                text, fr = "\n".join(f"{j}. बिंदु संख्या {j}" for j in range(1, 11)), "eos"  # 10 -> capped at 8
            elif i == 2:
                text, fr = "- एक\n- दो\n- तीन\n- चार अधूरा", "length"                        # truncated: last dropped
            else:
                text, fr = "रूपरेखा:\n* **मुख्य विचार** एक\n• दूसरा विचार\n१. तीसरा विचार\n\nअंत में नोट।", "eos"
            fake.append({**r, "text": text, "finish_reason": fr, "attempt": 0, "watermarked": False,
                         "model_id": "fake", "token_ids": [1, 2, 3]})
        # a second, better attempt for the first row: the lowest valid attempt wins
        fake.append({**ox[0], "text": "- क\n- ख\n- ग", "finish_reason": "eos", "attempt": 1, "watermarked": False})
        gen = Path(self.tmp.name) / "ox_gen.jsonl"
        gen.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in fake), encoding="utf-8")
        out = Path(self.tmp.name) / "O_prompts.jsonl"
        bep.main(["build-outline", "--gen", str(gen), "--out", str(out)])
        orows = {r["outline_source_id"]: r for r in read(out)}
        self.assertEqual(len(orows), len(ox))
        self.assertEqual(orows[ox[0]["id"]]["outline_attempt"], 1)
        self.assertEqual(orows[ox[1]["id"]]["n_bullets"], 8)
        self.assertEqual(orows[ox[2]["id"]]["n_bullets"], 3)
        if len(ox) > 3:
            self.assertEqual(orows[ox[3]["id"]]["points"], "- मुख्य विचार एक\n- दूसरा विचार\n- तीसरा विचार")
        ids = [r["id"] for r in orows.values()]
        self.assertEqual(len(ids), len(set(ids)))
        all_ids = {r["id"] for r in self.rows}
        for src in ox:
            o = orows[src["id"]]
            for k in REQUIRED:
                self.assertIn(k, o)
            self.assertEqual(o["level"], "O_outline")
            self.assertTrue(o["watermark"])
            self.assertNotIn("text", o)
            self.assertNotIn("token_ids", o)
            self.assertNotIn(o["id"], all_ids)
            self.assertEqual(o["id"], src["id"].replace("OX_extract_outline", "O_outline"))
            for f in ("pair_id", "experiment", "passage_id", "human_passage", "input_words", "input_tokens", "split"):
                self.assertEqual(o[f], src[f])
            self.assertEqual(o["cutoffs"], src["passage_cutoffs"])
            self.assertEqual(o["max_new_tokens"], ceil64(1.3 * src["input_tokens"]))
            self.assertEqual(o["target_words"], bep.round_to(src["input_words"], 10))
            user = o["messages"][1]["content"]
            self.assertIn(o["points"], user)
            self.assertIn(str(o["target_words"]), user)
            self.assertTrue(all(line.startswith("- ") for line in o["points"].splitlines()))
            self.assertRegex(o["template_id"], r"^O_outline/[012]$")
        # drop rule: one OX output with < 3 bullets and no second attempt
        bad = [{**ox[0], "text": "- केवल एक", "finish_reason": "eos", "attempt": 0}]
        rows, stats = bep.build_outline_rows(bad, self.cfg)
        self.assertEqual(rows, [])
        self.assertEqual(stats["dropped: < 3 bullets"], 1)


def _english_doc(i: int, n_sent: int, words_per_sent: int, genre: str) -> dict:
    sents = [" ".join(f"w{i}x{s}y{j}" for j in range(words_per_sent - 1)) + " end." for s in range(n_sent)]
    return {"id": f"d{i:03d}", "language": "en", "title": f"A Title Number {i}", "text": " ".join(sents),
            "genre": genre, "license": "CC0"}


class TestGenericJsonl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        docs = [_english_doc(i, 70, 20, ["story", "essay", "opinion"][i % 3]) for i in range(60)]
        # 1185 words in sentences of 15 -> words cut (1200) and tokens cut (1500 tok ~ 1190 w) coincide
        docs.append({**_english_doc(900, 79, 15, "story"), "id": "same"})
        cls.src = t / "medium_en.jsonl"
        cls.src.write_text("".join(json.dumps(d) + "\n" for d in docs), encoding="utf-8")
        cls.out = t / "out"
        run_build(cls.out, "--jsonl", str(cls.src), "--n-docs", "0", "--pilot-docs", "5")
        cls.rows = read(cls.out / "edit_prompts.jsonl")
        cls.passages = read(cls.out / "human_passages.jsonl")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fields_from_jsonl(self):
        r = self.rows[0]
        self.assertEqual(r["source"], "medium_en")
        self.assertTrue(r["pair_id"].startswith("medium_en/"))
        self.assertEqual(r["license"], "CC0")
        self.assertIn(r["genre"], ("story", "essay", "opinion"))
        self.assertEqual(r["messages"][0]["content"], bep.load_config()["kit_data"]["system"]["en"])
        self.assertEqual(Counter(p["split"] for p in self.passages if p["experiment"] == "same_words")["pilot"], 5)

    def test_both_experiments_and_dedupe(self):
        by_doc = defaultdict(list)
        for p in self.passages:
            by_doc[p["pair_id"]].append(p)
        self.assertEqual(len(by_doc), 61)
        same = by_doc["medium_en/same"]
        self.assertEqual(len(same), 1)
        self.assertEqual(same[0]["experiments"], ["same_words", "same_tokens"])
        self.assertEqual(same[0]["input_cut_by"], {"words": 1200, "tokens": 1500})
        other = by_doc["medium_en/d000"]
        self.assertEqual(sorted(p["experiment"] for p in other), ["same_tokens", "same_words"])
        sw = next(p for p in other if p["experiment"] == "same_words")
        st = next(p for p in other if p["experiment"] == "same_tokens")
        self.assertEqual((sw["n_words"], st["n_words"]), (1200, 1180))
        # 200 and 400 tokens fall between two sentence ends (176 < 190, 378 < 380 estimated tokens)
        self.assertEqual(sw["cutoffs"], {"words": [100, 200, 400, 800, 1200], "tokens": [800, 1500]})
        # L1 allowed for English (not punctuation-stripped); one set of rows for the deduped doc
        self.assertEqual(len([r for r in self.rows if r["pair_id"] == "medium_en/same" and r["level"] == "L1_proofread"]), 1)

    def test_templates_rotate_and_are_stable_per_doc(self):
        variants = defaultdict(Counter)
        per_doc = defaultdict(set)
        for r in self.rows:
            variants[r["level"]][r["template_id"]] += 1
            per_doc[(r["pair_id"], r["level"])].add(r["template_id"])
        for lv in ("L1_proofread", "L2_polish", "L3_paraphrase", "L4_rewrite", "C_continue"):
            self.assertEqual(len(variants[lv]), 3, (lv, variants[lv]))
            self.assertGreater(min(variants[lv].values()), len(self.passages) / 3 * 0.4)
        self.assertEqual(set(variants["G_title"]), {"G_title/0", "G_title/2"})
        # the same doc uses one phrasing per level in both experiments and at every k
        self.assertTrue(all(len(v) == 1 for v in per_doc.values()))

    def test_stratified_sample(self):
        out = Path(self.tmp.name) / "strat"
        run_build(out, "--jsonl", str(self.src), "--n-docs", "9", "--levels", "L3_paraphrase")
        rows = read(out / "edit_prompts.jsonl")
        self.assertEqual({r["level"] for r in rows}, {"L3_paraphrase"})
        genres = Counter(r["genre"] for r in rows if r["experiment"] == "same_words")
        self.assertEqual(genres, {"story": 3, "essay": 3, "opinion": 3})

    def test_levels_subset_adds_ox_for_o(self):
        self.assertEqual(bep.resolve_levels(bep.load_config(), ["O_outline"]), ["OX_extract_outline", "O_outline"])


@unittest.skipUnless(HAVE_CORPORA and (bep.DEFAULT_TOKENIZER / "tokenizer.json").exists(),
                     "staged Qwen3 tokenizer or corpora not available")
class TestRealTokenizer(unittest.TestCase):
    def test_token_cuts_with_qwen3(self):
        try:
            import tokenizers  # noqa: F401
        except ImportError:
            self.skipTest("pip install tokenizers")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            bep.main(["build", "--tokenizer", str(bep.DEFAULT_TOKENIZER), "--out", str(out), "--corpus", "samalochan",
                      "--n-docs", "3", "--levels", "L3_paraphrase"])
            passages = read(out / "human_passages.jsonl")
            rows = read(out / "edit_prompts.jsonl")
        from tokenizers import Tokenizer

        tk = Tokenizer.from_file(str(bep.DEFAULT_TOKENIZER / "tokenizer.json"))
        for p in passages:
            self.assertFalse(p["token_counts_estimated"])
            n = len(tk.encode(p["text"], add_special_tokens=False).ids)
            self.assertEqual(p["n_tokens"], n)
            if p["input_cutoff_kind"] == "tokens":
                self.assertTrue(0.95 * p["input_cutoff"] <= n <= p["input_cutoff"])
        for r in rows:
            self.assertEqual(r["max_new_tokens"], ceil64(1.5 * r["input_tokens"]))


class TestUnits(unittest.TestCase):
    def test_render_is_single_pass(self):
        self.assertEqual(bep.render("A {n} B\n\n{text}", n=5, text="keep {n} and {head}"), "A 5 B\n\nkeep {n} and {head}")
        with self.assertRaises(KeyError):
            bep.render("{points}", n=1)

    def test_line_breaks_are_sentence_ends_when_punctuation_removed(self):
        text = "पहला वाक्य यहाँ है  \n दूसरा वाक्य\n तीसरा वाक्य है"
        self.assertEqual(bep.sentence_ends(text, bep.SENT_END), [])
        ends = bep.sentence_ends(text, bep.LINE_END)
        self.assertEqual([text[:e].strip().split()[-1] for e in ends], ["है", "वाक्य", "है"])
        t = bep.cut_passage(text, "hi", "words", 6, None, 0.95, bep.LINE_END)
        self.assertEqual(t.passage, "पहला वाक्य यहाँ है  \n दूसरा वाक्य")
        self.assertTrue(t.accepted)

    def test_cut_rule(self):
        text = "One two three. Four five six seven. Eight nine ten."
        t = bep.cut_passage(text, "en", "words", 8, None)
        self.assertEqual(t.passage, "One two three. Four five six seven.")
        self.assertFalse(t.accepted)                       # 7 < 0.95 * 8
        self.assertTrue(bep.cut_passage(text, "en", "words", 7, None).accepted)
        self.assertEqual(bep.cut_passage(text, "en", "words", 2, None).reject_reason, "no_sentence_end")

    def test_exact_tokenizer_path(self):
        # fake "tokenizer": one token per 3 characters (monotone in the prefix, like BPE)
        fake = bep.TokenCounter(lambda s: math.ceil(len(s) / 3), {"en": 1.26, "hi": 4.57}, "fake")
        self.assertFalse(fake.estimated)
        doc = bep.SourceDoc("x/1", "x", "1", "en", "T", " ".join(f"Sentence {i} has a few words." for i in range(400)),
                            "unknown", "CC0", {"punctuation_removed": False})
        cfg = bep.load_config()
        ps, fails = bep.make_passages(doc, cfg, fake)
        self.assertFalse(fails)
        for p in ps:
            self.assertEqual(p.tokens, fake(p.text, "en"))
            if p.cutoff_kind == "tokens":
                self.assertTrue(0.95 * p.cutoff <= p.tokens <= p.cutoff)
            # truncation in cut_passage must not change the result
            full = bep.trim_text(doc.text, p.cutoff_kind, p.cutoff, fake.for_lang("en"), 0.95, bep.SENT_END)
            self.assertEqual(full.passage, p.text)

    def test_strip_header(self):
        text = "पेंटिग : लाल रत्नाकर\nरांगेय राघव\n_______________\nशीर्षक यहाँ\nलेखक नाम\nयह पहला वाक्य है. दूसरा."
        rest, n, hdr = bep.strip_header(text)
        self.assertEqual(rest, "यह पहला वाक्य है. दूसरा.")
        self.assertEqual(n, 5)
        story = "और तितली उड़ गई\nनूतन डिमरी गैरोला\n“बिंदुली?\nजी हां बिंदुली,"
        self.assertEqual(bep.strip_header(story)[0], "“बिंदुली?\nजी हां बिंदुली,")
        prose = "यह एक लंबी पंक्ति है जिसमें दस से अधिक शब्द हैं और यह गद्य का हिस्सा है.\nदूसरी"
        self.assertEqual(bep.strip_header(prose), (prose, 0, ""))

    def test_usable_title(self):
        chk = bep.load_config()["title_check"]
        self.assertTrue(bep.usable_title("सतपुड़ा के भीतर से", "hi", chk)[0])
        self.assertFalse(bep.usable_title("ravindra kalia", "hi", chk)[0])
        self.assertFalse(bep.usable_title("उर्दू से :", "hi", chk)[0])
        self.assertFalse(bep.usable_title("", "hi", chk)[0])
        self.assertFalse(bep.usable_title("यह एक वाक्य है.", "hi", chk)[0])
        self.assertTrue(bep.usable_title("The Long Walk Home", "en", chk)[0])

    def test_clean_outline(self):
        txt = "Here is the outline:\n\n1. First point\n2) **Second** point\n- third\nSome closing remark.\n* fourth"
        self.assertEqual(bep.clean_outline(txt), ["First point", "Second point", "third", "fourth"])
        self.assertEqual(bep.clean_outline(txt, truncated=True), ["First point", "Second point", "third"])
        self.assertEqual(len(bep.clean_outline("\n".join(f"- p{i}" for i in range(12)))), 8)


if __name__ == "__main__":
    unittest.main()
