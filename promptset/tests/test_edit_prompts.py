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
import validate_promptset as vp  # noqa: E402

HAVE_CORPORA = (REPO / "data" / "Samalochan").is_dir() and (REPO / "data" / "Hindi-Aesthetics-Corpus").is_dir()
REQUIRED = ("id", "pair_id", "language", "level", "genre", "source", "source_id", "license", "split", "experiment",
            "input_cutoff_kind", "input_cutoff", "input_words", "input_tokens", "human_passage", "k",
            "target_words", "max_new_tokens", "cutoffs", "template_id", "messages", "chat_template_kwargs",
            "generation", "watermark")
N_DOCS = 6


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


RND = int(bep.load_config()["lengths"]["max_new_tokens_round"])


def ceil_r(x: float) -> int:
    return int(math.ceil(x / RND) * RND)


def expected_mnt(r: dict, factor: float) -> int:
    """The level's factor x requested tokens, raised to what the served cutoffs need (validator rule)."""
    h = r.get("head_tokens") or 0
    t, w = r["input_tokens"], r["input_words"]
    need = [factor * (max(1, t - h) if r["level"] == "C_continue" else t)]
    need += [vp.MNT_CUTOFF_FACTOR * (n - h) for n in r["cutoffs"]["tokens"]]
    need += [n * t / w - h for n in r["cutoffs"]["words"]]
    return ceil_r(max(need))


def length_errors(rows: list[dict]) -> Counter:
    """validate_promptset's length checks on the rows: code -> count (errors only)."""
    rep = vp.Report()
    vp.check_lengths(rows, None, rep)
    return Counter({c: p["count"] for c, p in rep.problems.items() if p["severity"] == "error"})


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
            # samalochan: story/essay by its documented rule; hindi-aesthetics has no genre
            self.assertEqual(r["genre"] in ("story", "essay"), r["source"] == "samalochan", r["genre"])
            if r["source"] == "hindi-aesthetics":
                self.assertEqual(r["genre"], "unknown")
            self.assertTrue(r["license"])
            self.assertEqual(r["max_new_tokens"] % RND, 0)
            self.assertEqual(r["watermark"], r["level"] != "OX_extract_outline")
            self.assertNotRegex(r["messages"][1]["content"], r"\{(text|head|points|title|prompt|n)\}")
            if r["level"] != "C_continue":
                self.assertIsNone(r["k"])

    def test_max_new_tokens(self):
        for r in self.rows:
            lv, t = r["level"], r["input_tokens"]
            if lv.startswith("L"):
                self.assertEqual(r["max_new_tokens"], expected_mnt(r, 1.5))
                self.assertGreaterEqual(r["max_new_tokens"], 1.5 * t)
                self.assertEqual(r["target_words"], r["input_words"])
                self.assertIn(r["human_passage"], r["messages"][1]["content"])
            elif lv == "C_continue":
                self.assertEqual(r["max_new_tokens"], expected_mnt(r, 1.3))
                self.assertGreaterEqual(r["max_new_tokens"], 1.3 * (t - r["head_tokens"]))
            elif lv == "G_title":
                self.assertEqual(r["max_new_tokens"], expected_mnt(r, 1.3))
                self.assertIn(r["title"], r["messages"][1]["content"])
                self.assertIn(str(r["target_words"]), r["messages"][1]["content"])
                self.assertLessEqual(abs(r["target_words"] - r["input_words"]), 5)
            elif lv == "OX_extract_outline":
                self.assertEqual(r["max_new_tokens"], 1024)
                self.assertEqual(r["cutoffs"], {"words": [], "tokens": []})
                self.assertIn("passage_cutoffs", r)

    def test_budgets_pass_the_validator_length_checks(self):
        self.assertEqual(length_errors(self.rows), Counter())
        meta = json.loads((self.out / "build_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["script_min"], {"hi": 0.8, "en": 0.9})
        for p in self.passages:
            self.assertGreaterEqual(bep.script_share(p["text"], "hi"), 0.8, p["id"])

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
            thr = bep.min_units(n, L["min_frac"], bep.slack_for(L["slack"], kind, "hi"))
            self.assertGreaterEqual(units, thr, f"{p['id']}: 95% / slack rule")
            # largest cutoff the doc supports: no bigger cutoff of the same kind would be accepted
            bigger = [m for m in L["cutoffs"][kind] if m > n]
            for m in bigger:
                self.assertFalse(bep.cut_passage(body, "hi", kind, m, counter, L["min_frac"], pattern,
                                                 L["slack"]).accepted)
            # served cutoffs: exactly those at which the passage itself trims to an accepted prefix
            for k2 in ("words", "tokens"):
                for m in L["cutoffs"][k2]:
                    acc = bep.cut_passage(p["text"], "hi", k2, m, counter, L["min_frac"], pattern,
                                          L["slack"]).accepted
                    self.assertEqual(acc, m in p["cutoffs"][k2])
                    have = p["n_words"] if k2 == "words" else p["n_tokens"]
                    self.assertTrue(m <= max(have / 0.95, have + bep.slack_for(L["slack"], k2, "hi")) or not acc)
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
            self.assertEqual(o["max_new_tokens"], expected_mnt(o, 1.3))
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
        # 200 and 400 tokens fall between two sentence ends (176 and 378 estimated tokens): below 95 % (190, 380)
        # but within the en token slack of 38 (>= 162, 362), so they are served by default
        self.assertEqual(sw["cutoffs"], {"words": [100, 200, 400, 800, 1200], "tokens": [200, 400, 800, 1500]})
        # L1 allowed for English (not punctuation-stripped); one set of rows for the deduped doc
        self.assertEqual(len([r for r in self.rows if r["pair_id"] == "medium_en/same" and r["level"] == "L1_proofread"]), 1)

    def test_slack_recorded_and_slack_words_0_restores_pure_rule(self):
        meta = json.loads((self.out / "build_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["lengths"]["slack"], {"words": 30, "tokens": {"en": 38, "hi": 137}})
        self.assertIn("min(0.95 * N, N - slack)", meta["acceptance_rule"])
        self.assertIn("Length rule:", (self.out / "edit_prompts_summary.md").read_text(encoding="utf-8"))
        out = Path(self.tmp.name) / "noslack"
        run_build(out, "--jsonl", str(self.src), "--n-docs", "0", "--levels", "L3_paraphrase", "--slack-words", "0")
        meta0 = json.loads((out / "build_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta0["lengths"]["slack"], {"words": 0, "tokens": {"en": 0, "hi": 0}})
        self.assertIn("no slack", meta0["acceptance_rule"])
        sw = next(p for p in read(out / "human_passages.jsonl")
                  if p["pair_id"] == "medium_en/d000" and p["experiment"] == "same_words")
        self.assertEqual(sw["cutoffs"]["tokens"], [800, 1500])     # 176 < 190, 378 < 380: the old 95 % rule

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

    def test_budgets_pass_the_validator_length_checks(self):
        self.assertEqual(length_errors(self.rows), Counter())
        for r in self.rows:
            if r["level"] != "OX_extract_outline":
                self.assertEqual(r["cutoffs"], next(p["cutoffs"] for p in self.passages if p["id"] == r["passage_id"]))

    def test_levels_subset_adds_ox_for_o(self):
        self.assertEqual(bep.resolve_levels(bep.load_config(), ["O_outline"]), ["OX_extract_outline", "O_outline"])


# ----------------------------------------------------------------------------- study corpora on fixtures

def _hi_body(i: int, n_sent: int = 30) -> str:
    sents = [" ".join(f"शब्द{i}क{s}ख{j}" for j in range(11)) + " है।" for s in range(n_sent)]
    return "\n\n".join(" ".join(sents[k:k + 5]) for k in range(0, n_sent, 5))


def _en_body(i: int, n_sent: int = 30) -> str:
    sents = [" ".join(f"w{i}s{s}t{j}" for j in range(11)) + " end." for s in range(n_sent)]
    return "\n\n".join(" ".join(sents[k:k + 5]) for k in range(0, n_sent, 5))


def _write_title_author_dir(root: Path, rows: list[dict], fields: list[str]) -> None:
    """Files in the hindisamay / medium layout (title, blank, author, blank, paragraphs) plus metadata.csv."""
    import csv

    root.mkdir(parents=True)
    for r in rows:
        (root / r["file"]).write_text(f"{r['title']}\n\n{r[fields[1]]}\n\n{r.pop('_body')}", encoding="utf-8")
    with open(root / "metadata.csv", "w", encoding="utf-8-sig", newline="") as f:   # BOM, as HindiSamay's
        w = csv.DictWriter(f, ["file", "title", *fields[1:]])
        w.writeheader()
        w.writerows(rows)


def make_study_fixtures(base: Path, n_hs_story: int = 12, n_hs_essay: int = 8, n_sam_essay: int = 3,
                        n_med_story: int = 10, n_med_essay: int = 14) -> dict[str, Path]:
    hs = [{"file": f"{i:05d}_कहानी-{i}.txt", "title": f"पहली कहानी {i}", "author": f"लेखक {i}", "genre": "कहानी",
           "words": 0, "_body": _hi_body(i)} for i in range(n_hs_story)]
    fine = ["आलोचना", "लेख", "निबंध", "व्यंग्य"]
    hs += [{"file": f"{100 + i:05d}_लेख-{i}.txt", "title": f"विचार : एक लेख {i}", "author": f"लेखिका {i}",
            "genre": fine[i % 4], "words": 0, "_body": _hi_body(100 + i)} for i in range(n_hs_essay)]
    for r in hs:
        r["words"] = len(r["_body"].split())
    _write_title_author_dir(base / "HindiSamay", hs, ["title", "author", "genre", "words"])
    med = [{"file": f"2020-01-{i + 1:02d}_story-{i}.txt", "title": f"The Night Train: Part of No Series {i}",
            "authors": f"Author {i}", "genre": "story", "words": 0, "tags": "fiction; short story",
            "_body": _en_body(i)} for i in range(n_med_story)]
    med += [{"file": f"2021-02-{i + 1:02d}_essay-{i}.txt", "title": f"On Walking Home {i}", "authors": f"Writer {i}",
             "genre": "essay", "words": 0, "tags": "essay", "_body": _en_body(100 + i)} for i in range(n_med_essay)]
    for r in med:
        r["words"] = len(r["_body"].split())
    _write_title_author_dir(base / "Medium", med, ["title", "authors", "genre", "words", "tags"])
    sam = base / "Samalochan"
    sam.mkdir()
    for i in range(n_sam_essay):
        (sam / f"2015-0{i + 1}-01_आलेख-{i}.txt").write_text(f"समालोचना शीर्षक {i}\nआलोचक {i}\n" + _hi_body(200 + i),
                                                           encoding="utf-8")
    (sam / "2015-09-01_कथा-गाथा-कोई.txt").write_text("कहानी शीर्षक\nकथाकार\n" + _hi_body(300), encoding="utf-8")
    return {"hindisamay": base / "HindiSamay", "medium": base / "Medium", "samalochan": sam}


class StudyCorporaFixture(unittest.TestCase):
    """Points the registered hindisamay / medium / samalochan corpora at small fixture folders."""

    roots: dict[str, Path] = {}

    @classmethod
    def setUpClass(cls):
        import corpora
        from corpora import registry

        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name)
        cls.roots = make_study_fixtures(cls.base / "data")
        cls._saved = {n: registry._REGISTRY[n].default_root for n in cls.roots}
        for n, root in cls.roots.items():
            registry._REGISTRY[n].default_root = root
        cls.corpora = corpora

    @classmethod
    def tearDownClass(cls):
        from corpora import registry

        for n, root in cls._saved.items():
            registry._REGISTRY[n].default_root = root
        cls.tmp.cleanup()


class TestStudyCorporaLoading(StudyCorporaFixture):
    def test_hindisamay_and_medium_registered_with_metadata(self):
        hs = self.corpora.get("hindisamay")
        self.assertEqual(len(hs), 20)
        d = hs[0]
        self.assertEqual(d.title, "पहली कहानी 0")
        self.assertEqual((d.meta["genre"], d.meta["genre_fine"], d.meta["author"]), ("story", "कहानी", "लेखक 0"))
        self.assertEqual(d.meta["header_lines"], 2)
        self.assertIsInstance(d.meta["words"], int)
        self.assertEqual(Counter(x.meta["genre"] for x in hs), {"story": 12, "essay": 8})
        self.assertEqual({x.meta["genre_fine"] for x in hs if x.meta["genre"] == "essay"},
                         {"आलोचना", "लेख", "निबंध", "व्यंग्य"})
        med = self.corpora.get("medium")
        self.assertEqual(len(med), 24)                      # metadata.csv is not a document
        m = med[0]
        self.assertEqual((m.meta["genre"], m.meta["author"], m.meta["year"]), ("story", "Author 0", 2020))
        self.assertEqual(m.meta["tags"], ["fiction", "short story"])

    def test_samalochan_genre_rule(self):
        sam = {d.meta["slug"]: d for d in self.corpora.get("samalochan")}
        self.assertEqual(sam["कथा-गाथा-कोई"].meta["genre"], "story")
        self.assertEqual(sam["आलेख-0"].meta["genre"], "essay")
        from corpora.datasets import samalochan_genre

        self.assertEqual(samalochan_genre("ravindra", "पेंटिग : कोई\nकहानी\nज़िंदगी लावारिस\n..."),
                         ("story", "section label"))
        self.assertEqual(samalochan_genre("ravindra", "कहानी और राष्ट्र की परछाइयाँ\nअरुण देव\n")[0], "essay")

    def test_source_docs_strip_title_and_author(self):
        cfg = bep.load_config()
        est = bep.TokenCounter(None, cfg["lengths"]["tokens_per_word"])
        for name, lang in (("hindisamay", "hi"), ("medium", "en")):
            docs = bep.load_corpus_docs(name, cfg)
            self.assertEqual({d.language for d in docs}, {lang})
            self.assertTrue(all(d.genre in ("story", "essay") for d in docs))
            self.assertIn("do not redistribute", docs[0].license)
            ps, fails = bep.make_passages(docs[0], cfg, est)
            self.assertFalse(fails)
            for p in ps:
                self.assertEqual(p.header_lines_stripped, 2)
                self.assertFalse(p.text.startswith(docs[0].title))
                self.assertEqual(docs[0].text[p.char_start:p.char_start + len(p.text)], p.text)


class TestGenreMixSampling(StudyCorporaFixture):
    N = 10

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = cls.base / "mix"
        run_build(cls.out, "--n-docs", str(cls.N))             # no --corpus: the study build (genre_mix)
        cls.rows = read(cls.out / "edit_prompts.jsonl")
        cls.passages = read(cls.out / "human_passages.jsonl")
        cls.meta = json.loads((cls.out / "build_meta.json").read_text(encoding="utf-8"))

    def docs(self, passages=None) -> Counter:
        return Counter((lang, g, src) for lang, g, src, _ in
                       {(p["language"], p["genre"], p["source"], p["pair_id"]) for p in passages or self.passages})

    def test_same_genre_mix_per_language(self):
        self.assertEqual(self.meta["sampling_mode"], "genre_mix")
        got = self.docs()
        # 10 per language, 5 story + 5 essay; hi essays split hindisamay / samalochan (3 + 2 by largest remainder,
        # ties by name), hi stories from hindisamay only (samalochan's कथा-गाथा story is not used)
        self.assertEqual(got, {("en", "story", "medium"): 5, ("en", "essay", "medium"): 5,
                               ("hi", "story", "hindisamay"): 5, ("hi", "essay", "hindisamay"): 3,
                               ("hi", "essay", "samalochan"): 2})
        self.assertEqual(self.meta["genre_plan"]["hi"], {"story": {"hindisamay": 5},
                                                         "essay": {"hindisamay": 3, "samalochan": 2}})
        self.assertEqual(self.meta["docs"]["hi/story/samalochan"]["quota"], 0)
        self.assertNotIn("hindi-aesthetics", {r["source"] for r in self.rows})

    def test_english_rows(self):
        en = [r for r in self.rows if r["language"] == "en"]
        kit = bep.load_config()["kit_data"]
        self.assertTrue(en)
        self.assertEqual({r["source"] for r in en}, {"medium"})
        self.assertTrue({"L1_proofread", "L2_polish", "C_continue", "G_title", "OX_extract_outline"}
                        <= {r["level"] for r in en})
        for r in en:
            self.assertEqual(r["messages"][0]["content"], kit["system"]["en"])
            self.assertGreaterEqual(bep.script_share(r["messages"][1]["content"], "en"), 0.9)
            self.assertIn("do not redistribute", r["license"])
            if r["level"] == "OX_extract_outline":
                self.assertEqual(r["max_new_tokens"], 512)
            if r["level"] in ("L1_proofread", "L2_polish"):
                self.assertIn(r["human_passage"], r["messages"][1]["content"])
                self.assertFalse(r["human_passage"].startswith("The Night Train"))   # title line stripped
            if r["level"] == "G_title" and r["genre"] == "story":
                self.assertIn(":", r["title"])           # a colon subtitle is a usable metadata title
        self.assertEqual(length_errors(self.rows), Counter())
        self.assertIn(("en", "story"), {(r["language"], r["genre"]) for r in en})
        self.assertTrue(all(r["meta"].get("genre_fine") for r in self.rows if r["source"] != "samalochan"))
        self.assertTrue(all(r["meta"]["genre_rule"] for r in self.rows if r["source"] == "samalochan"))

    def test_balance_matches_across_languages(self):
        rep = vp.validate(self.rows)
        cells = [c for c in rep.sections["balance"]["cells"] if not c[0].startswith("G_title")]
        self.assertTrue(cells)
        self.assertEqual([c for c in cells if c[-1] != "ok"], [])
        self.assertEqual(rep.errors, 0)

    def test_summary_reports_coverage_and_budgets_per_language(self):
        md = (self.out / "edit_prompts_summary.md").read_text(encoding="utf-8")
        for h in ("Documents (genre-matched sample)", "Sampled documents per language x genre x source",
                  "Served-cutoff coverage per language", "max_new_tokens per language and split"):
            self.assertIn(h, md)
        tsv = (self.out / "titles_review.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(tsv[0].split("\t")[:4], ["pair_id", "source", "language", "genre"])
        self.assertEqual(len(tsv) - 1, 2 * self.N)

    def test_stable_and_growing_sample(self):
        out2 = self.base / "mix2"
        run_build(out2, "--n-docs", str(self.N))
        self.assertEqual((self.out / "edit_prompts.jsonl").read_bytes(), (out2 / "edit_prompts.jsonl").read_bytes())
        out3 = self.base / "mix3"
        run_build(out3, "--n-docs", str(self.N + 4), "--levels", "L1_proofread")
        big = {p["pair_id"] for p in read(out3 / "human_passages.jsonl")}
        self.assertLessEqual({p["pair_id"] for p in self.passages}, big)

    def test_short_source_is_topped_up_within_the_genre(self):
        out = self.base / "topup"
        run_build(out, "--n-docs", "16", "--levels", "L1_proofread")       # hi essays 8: samalochan has only 3
        got = self.docs(read(out / "human_passages.jsonl"))
        self.assertEqual(got[("hi", "essay", "samalochan")], 3)
        self.assertEqual(got[("hi", "essay", "hindisamay")], 5)
        self.assertEqual(got[("hi", "story", "hindisamay")], 8)

    def test_pilot_docs_per_language_keep_the_mix(self):
        out = self.base / "pilot"
        run_build(out, "--n-docs", str(self.N), "--pilot-docs", "4", "--levels", "L1_proofread")
        pil = {(p["language"], p["genre"], p["pair_id"]) for p in read(out / "human_passages.jsonl")
               if p["split"] == "pilot"}
        self.assertEqual(Counter((lang, g) for lang, g, _ in pil),
                         {("en", "story"): 2, ("en", "essay"): 2, ("hi", "story"): 2, ("hi", "essay"): 2})

    def test_named_corpora_default_to_per_corpus(self):
        out = self.base / "percorpus"
        run_build(out, "--corpus", "medium", "--n-docs", "4", "--levels", "L1_proofread")
        meta = json.loads((out / "build_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["sampling_mode"], "per_corpus")
        self.assertEqual(self.docs(read(out / "human_passages.jsonl")),
                         {("en", "story", "medium"): 2, ("en", "essay", "medium"): 2})
        out = self.base / "named_mix"
        run_build(out, "--corpus", "hindisamay", "--sampling", "genre_mix", "--n-docs", "6",
                  "--levels", "L1_proofread")
        self.assertEqual(self.docs(read(out / "human_passages.jsonl")),
                         {("hi", "story", "hindisamay"): 3, ("hi", "essay", "hindisamay"): 3})


class TestSamplingUnits(unittest.TestCase):
    def test_apportion(self):
        self.assertEqual(bep.apportion(300, {"story": 0.5, "essay": 0.5}), {"story": 150, "essay": 150})
        self.assertEqual(bep.apportion(5, {"hindisamay": 0.5, "samalochan": 0.5}), {"hindisamay": 3, "samalochan": 2})
        self.assertEqual(bep.apportion(7, {"a": 1, "b": 2, "c": 0}), {"a": 2, "b": 5, "c": 0})
        self.assertEqual(sum(bep.apportion(11, {"x": 0.3, "y": 0.3, "z": 0.4}).values()), 11)

    def test_genre_mix_plan_renormalises_over_loaded_sources(self):
        cfg = bep.load_config()
        self.assertEqual(cfg["sampling"]["genre_mix"], {"story": 0.5, "essay": 0.5})
        self.assertEqual(bep.genre_mix_plan(cfg, "hi", ["hindisamay", "samalochan"], 300),
                         {"story": {"hindisamay": 150}, "essay": {"hindisamay": 75, "samalochan": 75}})
        self.assertEqual(bep.genre_mix_plan(cfg, "hi", ["samalochan"], 10),
                         {"story": {"samalochan": 5}, "essay": {"samalochan": 5}})   # no share: equal split
        self.assertEqual(bep.genre_mix_plan(cfg, "en", ["medium"], 300),
                         {"story": {"medium": 150}, "essay": {"medium": 150}})

    def test_sample_genre_mix_order_and_unknown_genres(self):
        docs = [bep.SourceDoc(f"s/{i}", "s", str(i), "en", "", "x", g, "") for i, g in
                enumerate(["story"] * 5 + ["essay"] * 5 + ["poetry"] * 5)]
        chosen, counts = bep.sample_genre_mix(docs, {"story": {"s": 2}, "essay": {"s": 2}}, 1, lambda d: True)
        self.assertEqual([d.genre for d in chosen], ["story", "essay", "story", "essay"])
        self.assertEqual(counts[("poetry", "s")], {"available": 5, "quota": 0, "sampled": 0})
        # rejected docs are skipped and replaced, accept() is lazy (never called past the quota)
        seen = []
        chosen2, _ = bep.sample_genre_mix(docs, {"story": {"s": 2}, "essay": {"s": 0}}, 1,
                                          lambda d: seen.append(d.pair_id) or d.pair_id != "s/0")
        self.assertEqual(len(chosen2), 2)
        self.assertNotIn("s/0", [d.pair_id for d in chosen2])
        self.assertLessEqual(len(seen), 3)


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
                self.assertTrue(bep.min_units(p["input_cutoff"], 0.95, 137) <= n <= p["input_cutoff"])
        for r in rows:
            self.assertEqual(r["max_new_tokens"], expected_mnt(r, 1.5))
        self.assertEqual(length_errors(rows), Counter())


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

    def test_cut_rule_with_slack(self):
        text = " ".join(f"w{i}" for i in range(85)) + ". " + " ".join(f"v{i}" for i in range(30)) + "."
        L = bep.load_config()["lengths"]
        self.assertEqual(L["slack_words"], 30)
        self.assertEqual(L["slack"], {"words": 30, "tokens": {"en": 38, "hi": 137}})
        self.assertFalse(bep.cut_passage(text, "en", "words", 100, None, 0.95).accepted)        # no slack: 85 < 95
        t = bep.cut_passage(text, "en", "words", 100, None, 0.95, bep.SENT_END, L["slack"])
        self.assertTrue(t.accepted)
        self.assertEqual((t.n_words, t.min_units), (85, 70))
        self.assertTrue(bep.cut_passage(text, "en", "words", 100, None, 0.95, bep.SENT_END, 15).accepted)
        self.assertFalse(bep.cut_passage(text, "en", "words", 100, None, 0.95, bep.SENT_END, 4).accepted)
        # token slack is per language: 85 words -> 107 en / 388 hi estimated tokens
        est = bep.TokenCounter(None, L["tokens_per_word"])
        en = bep.cut_passage(text, "en", "tokens", 140, est, 0.95, bep.SENT_END, L["slack"])
        self.assertEqual((en.n_tokens, en.min_units, en.accepted), (107, 102, True))            # min(133, 140-38)
        self.assertFalse(bep.cut_passage(text, "en", "tokens", 140, est, 0.95).accepted)
        hi = bep.cut_passage(text, "hi", "tokens", 500, est, 0.95, bep.SENT_END, L["slack"])
        self.assertEqual((hi.n_tokens, hi.min_units, hi.accepted), (388, 363, True))            # min(475, 500-137)
        self.assertFalse(bep.largest_cut(text, "en", "words", [100], est, 0.95, bep.SENT_END) is not None)
        self.assertEqual(bep.largest_cut(text, "en", "words", [100], est, 0.95, bep.SENT_END, L["slack"])[0], 100)
        self.assertEqual(bep.served_cutoffs(text, "hi", {"words": [100], "tokens": [500]}, est, 0.95, bep.SENT_END,
                                            L["slack"]), {"words": [100], "tokens": [500]})
        self.assertEqual(bep.served_cutoffs(text, "hi", {"words": [100], "tokens": [500]}, est, 0.95, bep.SENT_END),
                         {"words": [], "tokens": []})

    def test_slack_overrides_in_config_and_cli(self):
        L = dict(bep.load_config()["lengths"])
        bep.set_slack(L, slack_words=10)
        self.assertEqual(L["slack"], {"words": 10, "tokens": {"en": 13, "hi": 46}})
        bep.set_slack(L, slack_tokens=50)
        self.assertEqual(L["slack"], {"words": 10, "tokens": 50})
        self.assertIn("tokens 50 (every language)", L["acceptance_rule"])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for name in ("promptset.yaml", "ai_levels_prompts.json"):
                (base / name).write_bytes((bep.CONFIG_DIR / name).read_bytes())
            cfg_text = (bep.CONFIG_DIR / "edit_levels.yaml").read_text(encoding="utf-8")
            cfg_text = cfg_text.replace("lengths:\n", "lengths:\n  slack_words: 0\n", 1)
            (base / "edit_levels.yaml").write_text(cfg_text, encoding="utf-8")
            L0 = bep.load_config(base / "edit_levels.yaml")["lengths"]
        self.assertEqual(L0["slack"], {"words": 0, "tokens": {"en": 0, "hi": 0}})
        self.assertIn("no slack", L0["acceptance_rule"])

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
                self.assertTrue(bep.min_units(p.cutoff, 0.95, 38) <= p.tokens <= p.cutoff)   # en token slack 38
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
        # known title + author header (hindisamay, medium): dropped even when the title is long, then the
        # heuristic still drops a short date line; a first body line with punctuation is kept
        long_title = "Why I Finally Stopped Pretending That I Enjoy Running Marathons Every Single Year"
        med = f"{long_title}\n\nJane Doe\n\nMay 4th, 2014\n\nI ran. Then I stopped.\n\nMore text here."
        self.assertEqual(bep.strip_header(med)[1], 0)                       # heuristic alone: title > 10 words
        rest, n, hdr = bep.strip_header(med, fixed_lines=2)
        self.assertEqual((rest.strip(), n), ("I ran. Then I stopped.\n\nMore text here.", 3))
        self.assertEqual([ln for ln in hdr.split("\n") if ln], [long_title, "Jane Doe", "May 4th, 2014"])
        self.assertEqual(bep.strip_header("T\n\nA\n\nBody one. Body two.", fixed_lines=2)[0].strip(),
                         "Body one. Body two.")

    def test_usable_title(self):
        chk = bep.load_config()["title_check"]
        self.assertTrue(bep.usable_title("सतपुड़ा के भीतर से", "hi", chk)[0])
        self.assertFalse(bep.usable_title("ravindra kalia", "hi", chk)[0])
        self.assertFalse(bep.usable_title("उर्दू से :", "hi", chk)[0])
        self.assertFalse(bep.usable_title("", "hi", chk)[0])
        self.assertFalse(bep.usable_title("यह एक वाक्य है.", "hi", chk)[0])
        self.assertTrue(bep.usable_title("The Long Walk Home", "en", chk)[0])
        self.assertTrue(bep.usable_title("और अन्ना सो रही थी...", "hi", chk)[0])          # trailing ellipsis
        self.assertFalse(bep.usable_title("Hide.", "en", chk)[0])
        # medium / hindisamay allow a colon inside a metadata title (subtitle), never a trailing one
        cfg = bep.load_config()
        med = {**chk, **bep.source_cfg(cfg, "medium")["title_check"]}
        self.assertFalse(bep.usable_title("The Alchemist: A book review", "en", chk)[0])
        self.assertTrue(bep.usable_title("The Alchemist: A book review", "en", med)[0])
        self.assertFalse(bep.usable_title("Mrs. Lovelock And The Iroquois Blanket:", "en", med)[0])
        self.assertTrue(bep.usable_title("कबीर : एक अंतहीन यात्रा", "hi",
                                         {**chk, **bep.source_cfg(cfg, "hindisamay")["title_check"]})[0])

    def test_cutoff_factor_matches_validator(self):
        self.assertEqual(bep.MNT_CUTOFF_FACTOR, vp.MNT_CUTOFF_FACTOR)
        self.assertEqual(bep.load_config()["lengths"]["max_context"], 32768)
        self.assertEqual(bep.passage_script_min(bep.load_config()), bep.DEFAULT_SCRIPT_MIN)
        self.assertEqual(bep.DEFAULT_SCRIPT_MIN["hi"], 0.8)

    def test_length_budget_covers_served_cutoffs(self):
        cuts = {"words": [100, 200], "tokens": [200, 400, 800, 1500]}
        # passage of 1400 tokens / 300 words serves 1500 tokens through the slack: 1.3 x 1400 = 1820 is not enough
        mnt, kept, dropped = bep.length_budget(1.3 * 1400, cuts, 1400, 300, 64)
        self.assertEqual((mnt, kept, dropped), (1920, cuts, 0))                 # ceil64(1.25 x 1500 = 1875)
        # continuation: the head counts towards the cutoff; k25 head of 350 tokens -> 1.25 x 1150 = 1437.5
        mnt, _, _ = bep.length_budget(1.3 * 1050, cuts, 1400, 300, 64, head_tokens=350)
        self.assertEqual(mnt, 1472)
        # a word cutoff above the passage (1200 words served from 1140) with a long head: 1.3 x (5244 - 4457)
        # = 1023 < 1200 x 4.6 tok/word - 4457 = 1063
        wc = {"words": [800, 1200], "tokens": [1500]}
        self.assertEqual(bep.length_budget(1.3 * (5244 - 4457), wc, 5244, 1140, 64, head_tokens=4457)[0], 1088)
        # the requested budget wins when it is larger
        self.assertEqual(bep.length_budget(5000, cuts, 1400, 300, 256)[0], 5120)
        # no cutoffs (or none above the passage): plain ceil of the request
        self.assertEqual(bep.length_budget(100.0, {"words": [], "tokens": []}, 80, 20, 64), (128, {"words": [], "tokens": []}, 0))

    def test_length_budget_drops_cutoffs_over_the_context_cap(self):
        cuts = {"words": [100, 200], "tokens": [200, 400, 800, 1500]}
        mnt, kept, dropped = bep.length_budget(600, cuts, 1400, 300, 64, cap=1100)
        self.assertEqual(kept, {"words": [100, 200], "tokens": [200, 400, 800]})   # 1.25 x 1500 > 1088
        self.assertEqual((mnt, dropped), (1024, 1))                                # ceil64(1.25 x 800) = 1024
        # request itself over the cap: capped at the cap (rounded down), unreachable cutoffs dropped
        mnt, kept, dropped = bep.length_budget(3000, cuts, 1400, 300, 64, cap=1100)
        self.assertEqual((mnt, dropped), (1088, 1))
        rep = vp.Report()
        vp.check_lengths([{"id": "x", "language": "hi", "max_new_tokens": mnt, "cutoffs": kept,
                           "input_tokens": 1400, "input_words": 300}], None, rep)
        self.assertEqual(rep.problems, {})

    def test_passage_script_check_skips_and_replaces_doc(self):
        hi_sent = "यह एक साधारण हिंदी वाक्य है जिसमें कुछ शब्द हैं।"
        latin_sent = "This sentence is mostly written in Latin script here ok."
        def doc(i, latin_every):
            sents = [latin_sent if latin_every and j % latin_every == 0 else hi_sent for j in range(120)]
            return {"id": f"h{i}", "language": "hi", "title": "शीर्षक", "text": " ".join(sents), "license": "CC0"}
        cfg = bep.load_config()
        est = bep.TokenCounter(None, cfg["lengths"]["tokens_per_word"])
        bad = bep.SourceDoc("t/x", "t", "x", "hi", "", doc(0, 2)["text"], "unknown", "CC0", {})
        ps, fails = bep.make_passages(bad, cfg, est)
        self.assertEqual(ps, [])
        self.assertTrue(all(why.startswith("script share") and why.endswith("< 0.8") for why in fails.values()), fails)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "hi_docs.jsonl"
            docs = [doc(i, 2 if i == 1 else 0) for i in range(5)]
            src.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
            full = Path(tmp) / "all"
            run_build(full, "--jsonl", str(src), "--n-docs", "0", "--levels", "L3_paraphrase")
            meta = json.loads((full / "build_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["docs_skipped_script"], 1)
            self.assertIn({"level": "(doc)", "source": "hi_docs", "n": 1,
                           "reason": "passage script share < 0.8: same_tokens; same_words"}, meta["skips"])
            self.assertIn("1 skipped", (full / "edit_prompts_summary.md").read_text(encoding="utf-8"))
            self.assertNotIn("hi_docs/h1", {p["pair_id"] for p in read(full / "human_passages.jsonl")})
            # a sample of n keeps n docs: the skipped one is replaced by the next eligible doc
            out = Path(tmp) / "n4"
            run_build(out, "--jsonl", str(src), "--n-docs", "4", "--levels", "L3_paraphrase")
            got = {p["pair_id"] for p in read(out / "human_passages.jsonl")}
            self.assertEqual(len(got), 4)
            self.assertNotIn("hi_docs/h1", got)
            # sampling.script_min overrides the threshold
            cfg2 = bep.load_config()
            cfg2["sampling"]["script_min"] = {"hi": 0.3}
            self.assertTrue(bep.make_passages(bad, cfg2, est)[0])

    def test_clean_outline(self):
        txt = "Here is the outline:\n\n1. First point\n2) **Second** point\n- third\nSome closing remark.\n* fourth"
        self.assertEqual(bep.clean_outline(txt), ["First point", "Second point", "third", "fourth"])
        self.assertEqual(bep.clean_outline(txt, truncated=True), ["First point", "Second point", "third"])
        self.assertEqual(len(bep.clean_outline("\n".join(f"- p{i}" for i in range(12)))), 8)


if __name__ == "__main__":
    unittest.main()
