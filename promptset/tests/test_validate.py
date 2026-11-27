"""Tests for scripts/validate_promptset.py: a small synthetic valid prompt set passes, and each rule fires on a
broken copy.  Run:  python -m pytest promptset/tests/test_validate.py -q

The Qwen3 test uses a real tokenizer dir from $PROMPTSET_TEST_TOKENIZER or
/mnt/project-files/promptset_stage/qwen3_tokenizer and is skipped when neither exists.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_promptset as vp  # noqa: E402
from build_promptset import render_user, seed_of  # noqa: E402

KIT = json.loads(vp.DEFAULT_KIT.read_text(encoding="utf-8"))
CFG = vp.load_yaml(vp.DEFAULT_CONFIG)
TEST_CFG = {**CFG, "genres": ["story", "essay"], "per_genre": {"main": 4, "reserve": 2}, "pilot_per_genre_tier": 1}

# (english, hindi) versions of a request; distinct subjects so nothing is a duplicate.
SUBJECTS = [
    ("an old lighthouse keeper", "एक बूढ़े दीपस्तंभ रखवाले"),
    ("a girl who builds a kite", "पतंग बनाने वाली एक लड़की"),
    ("a farmer waiting for rain", "बारिश का इंतज़ार करते एक किसान"),
    ("a lost dog in a big city", "बड़े शहर में खोए एक कुत्ते"),
    ("two brothers who run a tea stall", "चाय की दुकान चलाने वाले दो भाइयों"),
    ("a teacher in a mountain village", "पहाड़ी गाँव की एक अध्यापिका"),
    ("a boatman on a wide river", "एक चौड़ी नदी के नाविक"),
    ("a grandmother and her garden", "एक दादी और उनके बगीचे"),
    ("a clockmaker who never sleeps", "कभी न सोने वाले एक घड़ीसाज़"),
    ("a train that arrives every night", "हर रात आने वाली एक रेलगाड़ी"),
    ("a singer who loses her voice", "अपनी आवाज़ खो देने वाली एक गायिका"),
    ("a postman with one last letter", "आख़िरी चिट्ठी लिए एक डाकिये"),
]


def texts(i: int) -> dict[str, str]:
    en, hi = SUBJECTS[i]
    return {"en": f"Write about {en} and what changes for them over one long summer",
            "hi": f"{hi} के बारे में लिखिए और बताइए कि एक लंबी गर्मी में उनके जीवन में क्या बदलता है"}


def make_pair(i: int, genre: str, split: str, tier: str, native: str, source: str | None = None) -> list[dict]:
    source = source or ("wildchat" if native == "en" else "aya")
    t = texts(i)
    if source == "writingprompts":
        t = {"en": "A lighthouse keeper finds a message in a bottle that is addressed to him",
             "hi": "एक दीपस्तंभ रखवाले को बोतल में एक संदेश मिलता है जो उसी के नाम लिखा है"}
    rows = []
    for lang in ("en", "hi"):
        user, tid = render_user(KIT, lang, source, t[lang], CFG["tiers"][tier]["request_words"])
        r = {"id": f"G-{genre}-{i:04d}-{lang}", "pair_id": f"{source}:{i}", "language": lang, "level": "G",
             "genre": genre, "source": source, "source_id": str(i), "native_lang": native,
             "translated": lang != native, "translation_model": None if lang == native else "it2",
             "license": CFG["licenses"][source], "split": split, "length_tier": tier,
             "prompt_text": t[lang], "original_text": t[native], "template_id": tid,
             "messages": [{"role": "system", "content": KIT["system"][lang]}, {"role": "user", "content": user}],
             "chat_template_kwargs": {"enable_thinking": False}, "generation": {"seed": seed_of(f"x{i}{lang}")},
             "watermark": True}
        r.update(vp.expected_g_lengths(r, CFG))
        rows.append(r)
    return rows


def valid_rows() -> list[dict]:
    plan = [("pilot", "long", "en"), ("pilot", "short", "hi"), ("main", "long", "hi"), ("main", "short", "en"),
            ("reserve", "long", "en"), ("reserve", "short", "hi")]
    rows, i = [], 0
    for genre in ("story", "essay"):
        for split, tier, native in plan:
            src = "writingprompts" if (genre, split, tier) == ("story", "main", "short") else None
            rows += make_pair(i, genre, split, tier, native, src)
            i += 1
    return rows


def edit_rows() -> list[dict]:
    passages = {"en": "The rain came late that year, and the village waited on the riverbank for the first boats.",
                "hi": "उस साल बारिश देर से आई, और गाँव नदी किनारे पहली नावों का इंतज़ार करता रहा।"}
    rows = []
    for lang in ("en", "hi"):
        tmpl = next(lv for lv in KIT["levels"] if lv["id"] == "L1_proofread")[lang][0]
        rows.append({"id": f"L1-essay-0001-{lang}", "pair_id": "doc1", "language": lang, "level": "L1",
                     "experiment": "L1_proofread", "genre": "essay", "source": "samalochan", "license": "x",
                     "split": "main", "human_passage": passages[lang], "input_tokens": len(passages[lang]) // 4,
                     "max_new_tokens": 1024, "cutoffs": {"words": [], "tokens": [200]},
                     "messages": [{"role": "system", "content": KIT["system"][lang]},
                                  {"role": "user", "content": tmpl.replace("{text}", passages[lang])}],
                     "chat_template_kwargs": {"enable_thinking": False}, "generation": {"seed": 1}, "watermark": True})
    return rows


class FakeTok:
    """Qwen3-like rendering; 1 token per 4 characters. think=False mimics a template that ignores enable_thinking."""

    def __init__(self, think: bool = True):
        self.think = think

    def apply_chat_template(self, messages, add_generation_prompt, tokenize, enable_thinking):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        s += "<|im_start|>assistant\n"
        if self.think and enable_thinking is False:
            s += "<think>\n\n</think>\n\n"
        return s

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text) // 4))}


def run(rows, tok=None, cfg=TEST_CFG) -> vp.Report:
    return vp.validate(rows, KIT, cfg, tok)


def codes(rep: vp.Report, severity: str | None = None) -> set[str]:
    return {c for c, p in rep.problems.items() if severity is None or p["severity"] == severity}


def by_id(rows, rid):
    return next(r for r in rows if r["id"] == rid)


# ------------------------------------------------------------------------------------------- valid set


def test_valid_set_is_clean():
    rep = run(valid_rows() + edit_rows(), FakeTok())
    assert rep.problems == {}, rep.problems
    b = rep.sections["balance"]
    assert all(c[-1] == "ok" for c in b["cells"])
    assert {r[0]: r[4] for r in b["hindi_share"]} == {"essay": "0.50", "story": "0.50"}
    assert rep.sections["tokens"]["prompt_tokens"]


# ------------------------------------------------------------------------------------------- 1. schema


@pytest.mark.parametrize("field,value,code", [
    ("language", "ta", "schema.language"),
    ("max_new_tokens", 0, "schema.max_new_tokens"),
    ("max_new_tokens", True, "schema.max_new_tokens"),
    ("cutoffs", {"words": [100]}, "schema.cutoffs"),
    ("cutoffs", {"words": ["100"], "tokens": []}, "schema.cutoffs"),
    ("chat_template_kwargs", {"enable_thinking": True}, "schema.chat_template_kwargs"),
    ("chat_template_kwargs", {}, "schema.chat_template_kwargs"),
    ("generation", {"seed": "7"}, "schema.generation"),
    ("watermark", "yes", "schema.watermark"),
    ("split", "test", "schema.split"),
    ("license", "", "schema.license"),
    ("target_words", None, "schema.target_words"),
])
def test_schema_field(field, value, code):
    rows = valid_rows()
    rows[0][field] = value
    assert code in codes(run(rows), "error")


def test_schema_missing_field_and_messages():
    rows = valid_rows()
    del rows[0]["genre"]
    rows[2]["messages"] = rows[2]["messages"][1:]
    rows[4]["messages"][1]["content"] = "   "
    rows[6]["messages"] = list(reversed(rows[6]["messages"]))
    rep = run(rows)
    assert "schema.genre" in codes(rep)
    assert rep.problems["schema.messages"]["count"] == 3


def test_duplicate_ids_and_bad_json(tmp_path):
    rows = valid_rows()
    rows[1]["id"] = rows[0]["id"]
    assert "schema.id_duplicate" in codes(run(rows), "error")
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps(valid_rows()[0]) + "\n{not json\n[1]\n", encoding="utf-8")
    rep = vp.Report()
    assert len(vp.load_rows([p], rep)) == 1
    assert rep.problems["file.json"]["count"] == 2


def test_edit_fields_checked_only_when_present():
    rows = edit_rows()
    rows[0]["k"] = "50"
    rows[1]["input_tokens"] = -1
    rep = run(rows)
    assert {"schema.k", "schema.input_tokens"} <= codes(rep, "error")
    assert "schema.native_lang" not in codes(rep)  # G-only fields are not required of edit rows


# ------------------------------------------------------------------------------------------- 2. pairing


def test_pair_missing_language():
    rows = valid_rows()
    rows = [r for r in rows if r["id"] != "G-story-0000-hi"]
    rep = run(rows)
    assert "pair.count" in codes(rep, "error")
    assert "balance.en_hi" in codes(rep, "error")


def test_pair_mismatch_and_native():
    rows = valid_rows()
    by_id(rows, "G-story-0000-hi")["length_tier"] = "short"
    by_id(rows, "G-story-0001-en")["translated"] = False  # both rows untranslated
    rep = run(rows)
    assert rep.problems["pair.mismatch"]["examples"][0].startswith("wildchat:0")
    assert "pair.native" in codes(rep, "error")
    assert "pair.translated_flag" in codes(rep, "error")


def test_pair_native_lang_wrong():
    rows = valid_rows()
    for r in rows[:2]:
        r["native_lang"] = "hi"  # en row is untranslated but native_lang says hi
    assert "pair.native" in codes(run(rows), "error")


# ------------------------------------------------------------------------------------------- 3. balance


def test_balance_config_and_split_mix():
    rows = [r for r in valid_rows() if r["pair_id"] != "aya:1"]  # drop the Hindi-native pilot pair of story
    rep = run(rows)
    assert "balance.config" in codes(rep, "warning")
    # story pilot now only en-native; essay pilot still mixed -> the split overall is still mixed
    assert "balance.split_native_mix" not in codes(rep)
    rows = [r for r in valid_rows() if not (r["split"] == "pilot" and r["native_lang"] == "hi")]
    rep = run(rows)
    assert "balance.split_native_mix" in codes(rep, "warning")
    assert "balance.hindi_share" not in codes(rep)  # 2/5 = 0.4 is within 0.1 of 0.5
    rows = [r for r in valid_rows() if not (r["genre"] == "story" and r["native_lang"] == "hi")]
    rep = run(rows)
    assert rep.problems["balance.hindi_share"]["examples"] == ["story (0.00 vs 0.5)"]


def test_balance_tables_in_markdown():
    rep = run(valid_rows())
    md = vp.render_markdown(rep, ["G_prompts.jsonl"])
    assert "| G | pilot | story | long | 1 | 1 | ok |" in md
    assert "### Source mix per genre" in md and "| story | writingprompts | en | 1 |" in md


# ------------------------------------------------------------------------------------------- 4. script


def test_script_rules():
    rows = valid_rows()
    hi = by_id(rows, "G-story-0000-hi")
    hi["messages"][1]["content"] = "Write about a lighthouse keeper and his long summer. लगभग 1300 शब्दों में लिखें।"
    en = by_id(rows, "G-story-0001-en")
    en["messages"][1]["content"] += " धन्यवाद"
    by_id(rows, "G-essay-0006-hi")["messages"][0]["content"] = KIT["system"]["en"]
    rep = run(rows)
    assert rep.problems["script.hi_devanagari"]["examples"][0].startswith("G-story-0000-hi")
    assert rep.problems["script.en_devanagari"]["examples"][0].startswith("G-story-0001-en")
    assert rep.problems["script.system_prompt"]["examples"] == ["G-essay-0006-hi"]


def test_hindi_share_ignores_length_clause_and_counts_matras():
    assert vp.deva_share("लगभग 1300 शब्दों में लिखें।") == 1.0
    assert vp.deva_share("किताब book") == pytest.approx(5 / 9)


# ------------------------------------------------------------------------------------------- 5. content


def set_task(r: dict, task: str) -> None:
    r["prompt_text"] = task
    r["messages"][1]["content"], _ = render_user(KIT, r["language"], r["source"], task, r["target_words"])


@pytest.mark.parametrize("rid,task,code", [
    ("G-story-0000-en", "Write a story about a lighthouse keeper in 300 words or fewer please", "content.length_cap"),
    ("G-story-0000-en", "Write a poem about an old lighthouse keeper and the summer sea", "content.short_form"),
    ("G-story-0001-hi", "एक दीपस्तंभ रखवाले के बारे में संक्षेप में कहानी लिखिए", "content.length_cap"),
    ("G-story-0001-hi", "एक दीपस्तंभ रखवाले के बारे में एक कविता लिखिए", "content.short_form"),
    ("G-story-0000-en", "Write a story about the lighthouse at https://example.com/lh and its keeper", "content.url"),
    ("G-story-0000-en", "[ WP ] A lighthouse keeper finds a message in a bottle", "content.tag"),
    ("G-story-0000-en", "A lighthouse keeper finds a message <newline> in a bottle", "content.tag"),
    ("G-story-0000-en", "Write a story about {prompt} and the lighthouse keeper", "content.placeholder"),
    ("G-story-0000-en", "Write a story.", "content.too_short"),
])
def test_content_rules(rid, task, code):
    rows = valid_rows()
    set_task(by_id(rows, rid), task)
    rep = run(rows)
    assert code in codes(rep, "error"), rep.problems
    assert rep.problems[code]["examples"][0].startswith(rid)


def test_content_translated_soft_matches_are_warnings():
    rows = valid_rows()
    # en-native pair 0: its hi row is a translation; "छोटा सा" (small) and "शेर" (lion) are not real caps/forms
    set_task(by_id(rows, "G-story-0000-hi"), "एक छोटा सा शेर अपने जंगल में रास्ता खोजता है, उसकी कहानी लिखिए")
    rep = run(rows)
    assert {"content.length_cap_soft", "content.short_form_soft"} <= codes(rep, "warning")
    assert "content.length_cap" not in codes(rep)
    # ... but a numeric cap left in a translation is still an error
    set_task(by_id(rows, "G-story-0000-hi"), "एक दीपस्तंभ रखवाले की कहानी 300 शब्दों में लिखिए")
    assert "content.length_cap" in codes(run(rows), "error")


def test_length_clause():
    rows = valid_rows()
    r = by_id(rows, "G-story-0000-en")
    r["messages"][1]["content"] = r["messages"][1]["content"].replace("1300", "1200")
    r2 = by_id(rows, "G-essay-0007-hi")
    r2["messages"][1]["content"] += " लगभग 400 शब्दों में लिखें।"
    r3 = by_id(rows, "G-story-0002-en")
    r3["messages"][1]["content"] = r3["prompt_text"] + "."
    rep = run(rows)
    ex = rep.problems["content.length_clause"]["examples"]
    assert [e.split(" ")[0] for e in ex] == ["G-story-0000-en", "G-story-0002-en", "G-essay-0007-hi"] or \
        sorted(e.split(" ")[0] for e in ex) == sorted(["G-story-0000-en", "G-story-0002-en", "G-essay-0007-hi"])
    assert "content.render" in codes(rep, "warning")


def test_length_clause_premise_template_and_devanagari_digits():
    assert vp.clause_numbers("Write a story of about 400 words based on this idea: x", "en") == [400]
    assert vp.clause_numbers("इस विचार पर लगभग १३०० शब्दों की एक कहानी हिंदी में लिखें: x", "hi") == [1300]
    assert vp.clause_numbers("Write a 500-word essay. Write about 400 words.", "en") == [500, 400]


def test_duplicate_prompt_text():
    rows = valid_rows()
    a, b = by_id(rows, "G-story-0000-en"), by_id(rows, "G-essay-0006-en")
    set_task(b, a["prompt_text"].upper() + "!")
    rep = run(rows)
    assert rep.problems["content.duplicate"]["examples"][0].startswith("G-essay-0006-en")


# ------------------------------------------------------------------------------------------- 6. lengths


def test_max_new_tokens_rules():
    rows = valid_rows()
    a = by_id(rows, "G-story-0000-en")
    a["max_new_tokens"] = 1800  # < 1.25 x 1500
    b = by_id(rows, "G-story-0001-hi")
    b["cutoffs"] = {"words": [100, 200, 400, 800], "tokens": [200]}  # 800 x 4.57 > 2432
    rep = run(rows)
    assert rep.problems["length.mnt_vs_token_cutoff"]["examples"][0].startswith("G-story-0000-en")
    assert rep.problems["length.word_cutoff_unreachable"]["examples"][0].startswith("G-story-0001-hi")
    assert rep.problems["length.config"]["count"] == 2


def test_tokenizer_think_block_special_tokens_context():
    rows = valid_rows()
    rep = run(rows, FakeTok(think=False))
    assert rep.problems["tok.think_block"]["count"] == len(rows)
    rows = valid_rows()
    by_id(rows, "G-story-0000-en")["messages"][1]["content"] += " <think>plan</think>"
    rep = run(rows, FakeTok())
    assert "tok.special_in_content" in codes(rep, "error")
    assert rep.problems["tok.think_block"]["examples"][0].startswith("G-story-0000-en")
    rep = vp.validate(valid_rows(), KIT, TEST_CFG, FakeTok(), max_context=3000)
    assert "length.context" in codes(rep, "error")  # hi long: 7744 new tokens


def test_edit_rows_input_tokens_and_passage():
    rows = edit_rows()
    rows[0]["input_tokens"] += 40
    rows[1]["messages"][1]["content"] = rows[1]["messages"][1]["content"][:-20]
    rep = run(rows, FakeTok())
    assert rep.problems["tok.input_tokens"]["examples"][0].startswith("L1-essay-0001-en")
    assert rep.problems["edit.passage_in_prompt"]["examples"] == ["L1-essay-0001-hi"]


def test_no_tokenizer_given_is_noted():
    rep = vp.Report()
    assert vp.load_tokenizer(None, rep) is None and vp.load_tokenizer("/nonexistent/tok", rep) is None
    assert len(rep.notes) == 2


# ------------------------------------------------------------------------------------------- 7. translation


def test_translation_sanity():
    rows = valid_rows()
    set_task(by_id(rows, "G-story-0000-hi"), "दीपस्तंभ के रखवाले की कहानी लिखिए")  # 6 vs 13 words -> 0.46 ok
    set_task(by_id(rows, "G-story-0002-en"), "Write a story")  # hi-native 18 words -> 3 words
    set_task(by_id(rows, "G-essay-0006-hi"),
             "एक boatman और उसकी river के बारे में Ganga के किनारे की कहानी लिखिए और बताइए क्या बदलता है")
    set_task(by_id(rows, "G-essay-0007-en"),
             "Write about the the the the grandmother and her garden over one long summer season")
    rep = run(rows)
    assert rep.problems["tr.length_ratio"]["examples"][0].startswith("G-story-0002-en")
    assert rep.problems["tr.latin_leftover"]["examples"] == ["G-essay-0006-hi (boatman river)"]
    assert rep.problems["tr.repeat_loop"]["examples"][0].startswith("G-essay-0007-en")
    assert rep.sections["translation"]["capitalised_latin_words"] == 1
    assert vp.repeat_loop("घर जाओ घर जाओ घर जाओ अब") == "घर जाओ"
    assert vp.repeat_loop("very very very good") is None


def test_translation_checks_skip_native_rows():
    rows = valid_rows()
    set_task(by_id(rows, "G-story-0000-en"), "Write about the the the the lighthouse keeper and his summer")
    assert "tr.repeat_loop" not in codes(run(rows))


# ------------------------------------------------------------------------------------------- CLI


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_cli_strict_report_and_json(tmp_path, capsys):
    good = write_jsonl(tmp_path / "good.jsonl", valid_rows())
    edit = write_jsonl(tmp_path / "edit.jsonl", edit_rows())
    report = tmp_path / "out" / "report.md"
    # Real config: counts differ from per_genre (warnings only) -> strict still passes.
    assert vp.main([str(good), str(edit), "--report", str(report), "--strict"]) == 0
    summary = json.loads(report.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["errors"] == 0 and summary["warnings"] > 0
    assert "Tokenizer checks skipped" in report.read_text(encoding="utf-8")
    bad_rows = valid_rows()
    bad_rows[0]["chat_template_kwargs"]["enable_thinking"] = True
    bad = write_jsonl(tmp_path / "bad.jsonl", bad_rows)
    assert vp.main([str(bad), "--report", str(tmp_path / "bad.md"), "--strict"]) == 1
    assert vp.main([str(bad), "--report", str(tmp_path / "bad.md")]) == 0  # non-strict never fails
    assert "schema.chat_template_kwargs" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- real Qwen3


REAL_TOK = Path(os.environ.get("PROMPTSET_TEST_TOKENIZER", "/mnt/project-files/promptset_stage/qwen3_tokenizer"))


@pytest.mark.skipif(not REAL_TOK.is_dir(), reason="no local Qwen3 tokenizer")
def test_real_qwen3_tokenizer():
    rep = vp.Report()
    tok = vp.load_tokenizer(str(REAL_TOK), rep)
    if tok is None:
        pytest.skip("; ".join(rep.notes))
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    off = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    on = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
    assert off.endswith(vp.GEN_SUFFIX) and off == on + vp.THINK_EMPTY
    assert tok.convert_tokens_to_ids(["<think>", "</think>"]) == [151667, 151668]
    rows = valid_rows()
    for r in edit_rows():
        r["input_tokens"] = vp.n_tokens(tok, r["human_passage"])
        rows.append(r)
    rep = vp.validate(rows, KIT, TEST_CFG, tok)
    assert rep.problems == {}, rep.problems
    pt = {(lv, lang): (lo, hi) for lv, lang, _, lo, _, hi in rep.sections["tokens"]["prompt_tokens"]}
    assert pt[("G", "hi")][0] > pt[("G", "en")][0]  # Devanagari costs more tokens


def test_doubled_sentence_end_and_pipe_danda():
    rows = valid_rows()
    set_task(by_id(rows, "G-story-0000-en"), 'Write an essay arguing that "homework should be banned."')
    set_task(by_id(rows, "G-story-0001-hi"), "मेरे गाँव की सुबह पर एक निबंध लिखिए|")
    set_task(by_id(rows, "G-story-0002-en"), "Write about a quiet morning... and what the village hears then")
    rep = run(rows)
    ex = rep.problems["content.punct"]["examples"]
    # build_promptset.render_user no longer doubles the end after a closing quote, so only the
    # ASCII "|" used as a danda (left in prompt_text by set_task) is flagged.
    assert [e.split(" ")[0] for e in ex] == ["G-story-0001-hi"]


def continuation_row(lang: str = "en") -> dict:
    r = edit_rows()[0 if lang == "en" else 1]
    tmpl = next(lv for lv in KIT["levels"] if lv["id"] == "C_continue")[lang][0]
    head = r["human_passage"].split(",")[0] + ","
    r.update({"id": f"C_continue_k75-x-{lang}", "level": "C_continue", "experiment": "same_tokens", "k": 75,
              "head": head, "head_tokens": len(head) // 4, "input_words": 400, "input_tokens": 1000,
              "max_new_tokens": 320, "cutoffs": {"words": [], "tokens": [800]}})
    r["messages"][1]["content"] = tmpl.replace("{n}", "100").replace("{head}", head)
    r["input_tokens"] = len(r["human_passage"]) // 4
    return r


def test_continuation_rows_subtract_head_tokens():
    r = continuation_row()
    r["head_tokens"] = 600  # 1.25 x (800 - 600) = 250 <= 320: fine although 320 < 800
    rep = run([r])
    assert "length.mnt_vs_token_cutoff" not in codes(rep)
    assert "edit.passage_in_prompt" not in codes(rep)
    r["head_tokens"] = 400  # 1.25 x 400 = 500 > 320
    assert "length.mnt_vs_token_cutoff" in codes(run([r]), "error")


def test_edit_word_cutoff_uses_row_tokens_per_word():
    r = edit_rows()[1]  # hi
    r.update({"input_words": 1000, "input_tokens": 3000, "max_new_tokens": 4000,
              "cutoffs": {"words": [800], "tokens": [200]}})
    assert "length.word_cutoff_unreachable" not in codes(run([r]))  # 800 x 3.0 = 2400 (config 4.57 would be 3656)
    r["max_new_tokens"] = 2304
    assert "length.word_cutoff_unreachable" in codes(run([r]), "error")


def test_unwatermarked_outline_rows_may_serve_no_cutoff_and_edit_balance_is_a_warning():
    rows = edit_rows()
    rows[0].update({"level": "OX_extract_outline", "watermark": False, "cutoffs": {"words": [], "tokens": []}})
    rows[1]["cutoffs"] = {"words": [], "tokens": []}
    rep = run(rows)
    assert rep.problems["length.no_cutoffs"]["examples"] == ["L1-essay-0001-hi"]
    assert "balance.en_hi_edit" in codes(rep, "warning") and "balance.en_hi" not in codes(rep)


def test_edit_balance_counts_a_row_in_every_experiment_it_serves():
    en, hi = edit_rows()
    en.update({"experiment": "same_words", "experiments": ["same_words", "same_tokens"]})   # cuts coincide
    hi_w = {**hi, "experiment": "same_words", "experiments": ["same_words"]}
    hi_t = {**hi, "id": hi["id"] + "-st", "experiment": "same_tokens", "experiments": ["same_tokens"]}
    rep = run([en, hi_w, hi_t])
    assert "balance.en_hi_edit" not in codes(rep)
    cells = {tuple(c[:4]): c[4:] for c in rep.sections["balance"]["cells"]}
    assert cells[("L1", "main", "essay", "same_tokens")] == [1, 1, "ok"]
    rep = run([{**en, "experiments": ["same_words"]}, hi_w, hi_t])
    assert "balance.en_hi_edit" in codes(rep, "warning")


def test_head_tokens_checked_and_estimates_skipped():
    r = continuation_row("hi")
    r["head_tokens"] += 30
    rep = run([r], FakeTok())
    assert rep.problems["tok.input_tokens"]["examples"][0].startswith("C_continue_k75-x-hi (head_tokens=")
    r["token_counts_estimated"] = True
    rep = run([r], FakeTok())
    assert "tok.input_tokens" not in codes(rep) and any("token_counts_estimated" in n for n in rep.notes)


def test_title_rows_probe_the_title():
    r = edit_rows()[0]
    tmpl = next(lv for lv in KIT["levels"] if lv["id"] == "G_generate")["en"][0]
    r.update({"level": "G_title", "title": "Monsoon", "target_words": 300})
    r["messages"][1]["content"] = tmpl.replace("{n}", "300").replace("{title}", "Monsoon")
    assert "edit.passage_in_prompt" not in codes(run([r]))
    r["title"] = "Winter"
    assert "edit.passage_in_prompt" in codes(run([r]), "warning")
