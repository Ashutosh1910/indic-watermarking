"""Unit tests for the trimming rule, sentence ends, cleaning and the script check (no model needed)."""

from __future__ import annotations

import json

import pytest

import trim
from trim import SENT_END as SENT_END_DEFAULT
from trim import Options, clean_text, process, script_share, sentence_ends, trim_text


def ends_text(text: str) -> list[str]:
    """The text up to each sentence end."""
    return [text[:e] for e in sentence_ends(text)]


def one_token_per_char(s: str) -> int:
    """A stand-in token counter that differs from word counts (1 token per non-space char)."""
    return sum(1 for c in s if not c.isspace())


def words(n: int, start: int = 0) -> str:
    return " ".join(f"w{i}" for i in range(start, start + n))


# --------------------------------------------------------------------------- sentence ends


def test_basic_english_ends():
    t = "One. Two! Three? Four"
    assert ends_text(t) == ["One.", "One. Two!", "One. Two! Three?"]


def test_closing_quotes_and_brackets_after_punctuation():
    t = 'He said, "Stop!" Then: “Go.” (Really?) [Yes.] «Oui.» ‘No.’ \'Ok.\' end'
    got = [t[:e] for e in sentence_ends(t)]
    tails = [g.rsplit(" ", 1)[-1] for g in got]
    assert tails == ['"Stop!"', "“Go.”", "(Really?)", "[Yes.]", "«Oui.»", "‘No.’", "'Ok.'"]


def test_combined_quote_bracket_and_end_of_text():
    t = 'She whispered ("it is over.")'
    assert sentence_ends(t) == [len(t)]


def test_devanagari_danda_and_double_danda():
    t = "वह घर गया। फिर लौटा॥ क्या? हाँ ।अंत"
    got = [t[:e] for e in sentence_ends(t)]
    assert got[0].endswith("गया।") and got[1].endswith("लौटा॥") and got[2].endswith("क्या?")
    # "हाँ ।अंत": danda not followed by whitespace -> not an end
    assert len(got) == 3


def test_danda_with_space_before_and_quote_after():
    t = 'उसने कहा, "चलो ।" फिर'
    assert [t[:e] for e in sentence_ends(t)] == ['उसने कहा, "चलो ।"']


def test_ellipsis_counts_as_end():
    t = "He waited... and waited… Then left"
    assert ends_text(t) == ["He waited...", "He waited... and waited…"]


def test_decimal_and_inner_dots_are_not_ends():
    t = "It cost 3.5 rupees at www.example.com today"
    assert sentence_ends(t) == []


def test_paragraph_break_without_punctuation_is_not_an_end():
    t = "A heading without punctuation\n\nThe body ends here.\n\nA trailing line"
    got = [t[:e] for e in sentence_ends(t)]
    assert got == ["A heading without punctuation\n\nThe body ends here."]


def test_ascii_danda_option():
    t = "वह घर गया | फिर लौटा"
    assert sentence_ends(t) == []
    assert len(sentence_ends(t, trim._sent_end_re(ascii_danda=True))) == 1


# --------------------------------------------------------------------------- trimming rule


def test_trim_to_last_sentence_end_at_or_before_cutoff():
    text = f"{words(10)}. {words(9, 10)}. {words(10, 19)}."
    r = trim_text(text, "words", 20, None)
    assert r.passage == f"{words(10)}. {words(9, 10)}." and r.n_words == 19
    assert r.accepted  # 19 >= 0.95 * 20
    r = trim_text(text, "words", 21, None)
    assert r.n_words == 19 and not r.accepted  # 19 < 0.95 * 21 = 19.95


def test_end_exactly_at_cutoff_is_included():
    text = f"{words(20)}. {words(5, 20)}."
    r = trim_text(text, "words", 20, None)
    assert r.n_words == 20 and r.accepted


def test_95_percent_rule_boundary():
    text = f"{words(19)}. {words(10, 19)}."
    assert trim_text(text, "words", 20, None).accepted            # 19 >= 19.0
    text = f"{words(18)}. {words(10, 18)}."
    r = trim_text(text, "words", 20, None)
    assert not r.accepted and r.reject_reason == "too_short" and r.n_words == 18
    assert trim_text(text, "words", 20, None, min_frac=0.9).accepted


def test_text_shorter_than_cutoff():
    r = trim_text(f"{words(50)}.", "words", 100, None)
    assert not r.accepted and r.reject_reason == "too_short" and r.full_units == 50
    r = trim_text(f"{words(96)}.", "words", 100, None)
    assert r.accepted and r.n_words == 96  # shorter than N but >= 95 %


def test_unfinished_last_sentence_is_dropped():
    r = trim_text(f"{words(98)}. and then the text stops mid", "words", 200, None)
    assert r.passage.endswith("w97.") and r.n_words == 98


def test_no_sentence_end_and_empty():
    r = trim_text(words(30), "words", 20, None)
    assert not r.accepted and r.reject_reason == "no_sentence_end" and r.passage == ""
    r = trim_text("   \n ", "words", 20, None)
    assert r.reject_reason == "empty"
    # the only sentence end lies beyond the cutoff
    r = trim_text(f"{words(30)}.", "words", 20, None)
    assert r.reject_reason == "no_sentence_end"


def test_word_vs_token_cutoffs_differ():
    # 10 words of 2 chars ("w0".."w9") per sentence: 10 words, 20 "tokens" (+1 for the period)
    text = f"{words(10)}. {words(10, 10)}. {words(10, 20)}."
    by_words = trim_text(text, "words", 20, None)
    by_tokens = trim_text(text, "tokens", 45, one_token_per_char)
    assert by_words.n_words == 20 and by_words.accepted
    assert by_tokens.n_tokens == one_token_per_char(by_tokens.passage) <= 45
    assert by_tokens.n_words == 10 and not by_tokens.accepted  # 21 tokens < 0.95 * 45
    assert trim_text(text, "tokens", 22, one_token_per_char).accepted


def test_token_cutoff_with_real_tokenizer(tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(tiny_model_dir))

    def count(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    from conftest import HI_TEXT
    text = " ".join(HI_TEXT.split())
    for n in (20, 40, 80):
        r = trim_text(text, "tokens", n, count)
        assert r.n_tokens == count(r.passage) <= n
        assert r.passage.endswith(("।", "!", "?", "॥", '"')) or not r.passage
        # no later sentence end fits
        later = [e for e in sentence_ends(text) if e > len(r.passage)]
        if later:
            assert count(text[:later[0]].strip()) > n


def test_token_cutoff_needs_counter():
    with pytest.raises(ValueError):
        trim_text("A b.", "tokens", 5, None)


# --------------------------------------------------------------------------- script check


def test_script_share_counts_letters_and_vowel_signs():
    assert script_share("नमस्ते दुनिया।", "hi") == 1.0           # matras/virama count as Devanagari
    assert script_share("Hello, world! 123", "en") == 1.0         # digits/punctuation ignored
    mixed = "यह एक test sentence है।"
    assert 0.2 < script_share(mixed, "hi") < 0.8
    assert script_share("12345 ...", "hi") == 0.0
    assert script_share("abc", "ta") is None


def test_script_rejection_applies_after_length():
    opt = Options(mode="human", words=[10])
    hindi_in_latin = "Yah ek kahani hai jo bahut purani hai aur sab log jaante hain."
    t, share, ok = trim.evaluate(hindi_in_latin, "hi", "words", 13, None, opt)
    assert not t.accepted and t.reject_reason == "script" and ok is False and share == 0.0
    t, share, ok = trim.evaluate("यह एक कहानी है जो बहुत पुरानी है और सब लोग इसे जानते हैं।", "hi", "words", 14, None, opt)
    assert t.accepted and ok is True


# --------------------------------------------------------------------------- cleaning


@pytest.mark.parametrize("pre", [
    "Sure, here is the story you asked for:",
    "Certainly!",
    "Here's a story about a lighthouse keeper.",
    "Of course! Here is the polished version:",
    "**Here is the rewritten text:**",
    "ज़रूर! यह रही आपकी कहानी:",
    "जरूर, यह रहा संशोधित पाठ:",
    "प्रस्तुत है कहानी का अगला भाग:",
    "यहाँ आपकी कहानी की रूपरेखा है:",
    "यहां संशोधित संस्करण है:",
])
def test_preamble_line_is_dropped(pre):
    c = clean_text(f"{pre}\n\nThe sea was calm. वह घर गया।")
    assert c.text == "The sea was calm. वह घर गया।"
    assert "preamble" in c.steps and c.removed["preamble"] == pre


def test_preamble_with_precomposed_and_decomposed_nukta():
    for za in ("ज़", "ज़"):
        c = clean_text(f"{za}रूर!\n\nकहानी शुरू होती है।")
        assert c.text == "कहानी शुरू होती है।", za


@pytest.mark.parametrize("first", [
    "Here is where my grandmother lived for forty years.",
    "Sure enough, the boat never came back.",
    "Of course nobody believed him when he said the sea had spoken.",
    "यहाँ हर कोई एक दूसरे को जानता था।",
])
def test_genuine_first_line_is_kept(first):
    c = clean_text(f"{first}\nThe rest of the story follows.")
    assert c.text.startswith(first) and "preamble" not in c.steps


def test_two_preamble_lines():
    c = clean_text("Certainly!\nHere is your story:\n\nIt began at dawn.")
    assert c.text == "It began at dawn." and c.removed["preamble"] == "Certainly!\nHere is your story:"


@pytest.mark.parametrize("title", ["# The Lighthouse", "## समुद्र किनारे", "**The Lighthouse**",
                                   "Title: The Lighthouse", "शीर्षक: दीपस्तंभ", "The Last Lighthouse",
                                   "आख़िरी दीपस्तंभ"])
def test_title_line_is_dropped(title):
    c = clean_text(f"{title}\n\nThe keeper woke early. He climbed the stairs.")
    assert c.text == "The keeper woke early. He climbed the stairs." and c.removed["title"] == title


def test_short_first_line_kept_without_blank_line_or_with_punctuation():
    assert "title" not in clean_text("The Last Lighthouse\nstood on the cliff.").steps
    assert "title" not in clean_text("It was dawn.\n\nThe keeper woke.").steps
    long_line = " ".join(["word"] * 13)
    assert "title" not in clean_text(f"{long_line}\n\nThe keeper woke.").steps


def test_preamble_then_title_then_markdown():
    c = clean_text("  Sure, here is the story:\n\n# The Lighthouse\n\nThe **keeper** woke __early__.  ")
    assert c.text == "The keeper woke early."
    assert c.steps == ["strip", "preamble", "title", "markdown"]


@pytest.mark.parametrize("notes", [
    "Note: This story is about 1,300 words.",
    "**Note:** I kept the original tone.",
    "Word count: 1298",
    "(Word count: ~1300)",
    "---\n\nI hope you enjoy this story! Let me know if you want changes.",
    "नोट: कहानी लगभग 1300 शब्दों की है।",
    "शब्द संख्या: 1290",
])
def test_trailing_notes_are_dropped(notes):
    body = "The keeper woke early.\n\nHe climbed the stairs."
    c = clean_text(f"{body}\n\n{notes}")
    assert c.text == body and "trailing_notes" in c.steps


def test_scene_break_with_long_tail_is_kept():
    tail = " ".join(["word"] * 150) + "."
    text = f"Part one ends here.\n\n---\n\n{tail}"
    c = clean_text(text)
    assert c.text == text and "trailing_notes" not in c.steps


def test_cleaning_steps_can_be_switched_off():
    text = "Sure, here is the story:\n\n# Title\n\nThe **keeper** woke.\n\nNote: done."
    c = clean_text(text, steps=[s for s in trim.CLEAN_STEPS if s not in ("preamble", "title", "markdown")])
    assert c.text.startswith("Sure, here is the story:") and "**keeper**" in c.text
    assert "Note:" not in c.text
    assert clean_text(text, steps=[]).text == text


# --------------------------------------------------------------------------- rows, attempts, regeneration


def gen_row(rid: str, attempt: int, text: str, cutoffs: dict | None = None, language: str = "en") -> dict:
    return {"id": rid, "pair_id": rid.split("-", 1)[-1], "language": language, "level": "G", "genre": "story",
            "source": "test", "cutoffs": cutoffs or {"words": [10, 20]}, "attempt": attempt, "watermarked": True,
            "text": text, "token_ids": [1, 2, 3], "messages": [], "split": "main"}


def test_first_accepted_attempt_per_cutoff_and_regen_ids():
    short = f"{words(10)}. {words(3, 10)}"                 # ok for 10, too short for 20
    long_ = f"{words(10)}. {words(10, 10)}. {words(5, 20)}."  # ok for 10 and 20
    rows = [gen_row("en-a", 0, short), gen_row("en-a", 1, long_),
            gen_row("en-b", 0, short), gen_row("en-b", 1, short)]
    opt = Options(mode="gen")
    passages, cleaned, regen, no_cut = process(rows, opt, None)
    by = {(p["id"], p["cutoff"]): p for p in passages}
    assert by[("en-a", 10)]["attempt"] == 0 and by[("en-a", 10)]["accepted"]
    assert by[("en-a", 20)]["attempt"] == 1 and by[("en-a", 20)]["accepted"]
    assert by[("en-b", 20)]["attempt"] == 1 and not by[("en-b", 20)]["accepted"]
    assert by[("en-b", 20)]["attempts_tried"] == [0, 1]
    assert regen == {("en-b", True): 2}
    assert len(cleaned) == 4 and no_cut == []
    p = by[("en-a", 10)]
    assert "messages" not in p and "token_ids" not in p and "text" not in p and p["split"] == "main"


def test_cli_overrides_row_cutoffs_and_rows_without_cutoffs():
    rows = [gen_row("en-a", 0, f"{words(10)}."), {**gen_row("en-o", 0, "x."), "cutoffs": None}]
    passages, _, _, no_cut = process(rows, Options(mode="gen", words=[10]), None)
    assert [(p["id"], p["cutoff"]) for p in passages] == [("en-a", 10), ("en-o", 10)]
    passages, _, _, no_cut = process(rows, Options(mode="gen"), None)
    assert no_cut == ["en-o"]


def test_main_human_and_gen_mode(tmp_path):
    human = tmp_path / "human.jsonl"
    human.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in [
        {"id": "h1", "language": "hi", "text": "वह अपने घर गया। " * 30, "corpus": "test"},
        {"id": "h2", "language": "en", "text": f"{words(15)}.", "corpus": "test"},
    ]) + "\n", encoding="utf-8")
    out = tmp_path / "trim" / "human_trim.jsonl"
    assert trim.main(["human", "--in", str(human), "--out", str(out), "--tokenizer", "none",
                      "--words", "20", "60"]) == 0
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4 and {r["corpus"] for r in rows} == {"test"}
    by = {(r["id"], r["cutoff"]): r for r in rows}
    assert by[("h1", 20)]["accepted"] and by[("h1", 20)]["n_words"] == 20
    assert by[("h1", 60)]["accepted"] and by[("h1", 60)]["script_ok"] is True
    assert by[("h2", 20)]["reject_reason"] == "too_short" and by[("h2", 20)]["level"] == "human"
    summary = json.loads((tmp_path / "trim" / "human_trim.summary.json").read_text())
    assert summary["meta"]["n_passages"] == 4
    assert (tmp_path / "trim" / "human_trim.summary.md").exists()

    gen = tmp_path / "gen.jsonl"
    gen.write_text(json.dumps(gen_row("en-a", 0, "Sure, here is the story:\n\n" + f"{words(10)}. {words(3)}")) + "\n")
    out = tmp_path / "trim" / "gen_trim.jsonl"
    assert trim.main(["gen", "--in", str(gen), "--out", str(out), "--tokenizer", "none"]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["accepted"] for r in rows] == [True, False] and rows[0]["cleaning"] == ["preamble"]
    assert (tmp_path / "trim" / "gen_trim.regen_ids.txt").read_text() == "en-a\n"
    assert (tmp_path / "trim" / "gen_trim.regen_ids.attempt1.txt").read_text() == "en-a\n"
    cleaned = [json.loads(l) for l in (tmp_path / "trim" / "gen_trim.cleaned.jsonl").read_text().splitlines()]
    assert cleaned[0]["text_raw"].startswith("Sure") and not cleaned[0]["text_clean"].startswith("Sure")


# --------------------------------------------------------------------------- sentence_end_rule


def test_linebreak_rule_counts_line_ends():
    pat = trim.sentence_end_pattern("punct+linebreak")
    t = "पहली पंक्ति बिना विराम\nदूसरी पंक्ति भी  \n\nतीसरी पंक्ति। चौथी"
    got = [t[:e] for e in sentence_ends(t, pat)]
    assert got == ["पहली पंक्ति बिना विराम", "पहली पंक्ति बिना विराम\nदूसरी पंक्ति भी",
                   "पहली पंक्ति बिना विराम\nदूसरी पंक्ति भी  \n\nतीसरी पंक्ति।", t]
    assert len(sentence_ends(t)) == 1  # the default "punct" rule sees only the danda
    with pytest.raises(ValueError):
        trim.sentence_end_pattern("lines")


def test_linebreak_rule_matches_edit_builder():
    try:
        import build_edit_prompts
    except Exception as e:  # pragma: no cover
        pytest.skip(f"build_edit_prompts not importable: {e}")
    assert trim.LINE_END.pattern == build_edit_prompts.LINE_END.pattern


def test_rows_honour_sentence_end_rule_in_human_and_gen_mode():
    lines = "\n".join(" ".join(f"श{i}{j}" for j in range(5)) for i in range(10))  # 10 lines x 5 words, no punct
    human = {"id": "h", "language": "hi", "text": lines, "sentence_end_rule": "punct+linebreak"}
    passages, *_ = process([human], Options(mode="human", words=[20]), None)
    assert passages[0]["accepted"] and passages[0]["n_words"] == 20
    assert passages[0]["sentence_end_rule"] == "punct+linebreak"
    passages, *_ = process([{**human, "sentence_end_rule": "punct"}], Options(mode="human", words=[20]), None)
    assert passages[0]["reject_reason"] == "no_sentence_end"
    # gen mode: the short-line title heuristic is off for these rows, so the first line stays
    g = {**gen_row("hi-x", 0, lines.replace("\n", "\n\n", 1), {"words": [20]}, "hi"),
         "sentence_end_rule": "punct+linebreak"}
    passages, cleaned, *_ = process([g], Options(mode="gen"), None)
    assert passages[0]["accepted"] and passages[0]["passage"].startswith("श00") and "title" not in cleaned[0]["cleaning"]
    with pytest.raises(ValueError):
        process([{**human, "sentence_end_rule": "bogus"}], Options(mode="human", words=[20]), None)


# --------------------------------------------------------------------------- C_continue joining


def c_row(head: str, cont: str, rule: str = "punct", **kw) -> dict:
    return {**gen_row("C_continue_k50-x-en", 0, cont, kw.pop("cutoffs", {"words": [10, 20, 26]}),
                      kw.pop("language", "en")),
            "level": "C_continue", "k": 50, "head": head, "sentence_end_rule": rule, **kw}


def test_continue_rows_join_head_and_record_ai_share():
    head = f"{words(10)}. {words(5, 10)}."                       # 15 human words
    cont = "Sure, here is the continuation:\n\n" + f"{words(10, 100)}. {words(10, 200)}."
    passages, cleaned, *_ = process([c_row(head, cont)], Options(mode="gen"), one_token_per_char)
    by = {p["cutoff"]: p for p in passages}
    assert cleaned[0]["ai_char_start"] == len(head) + 1 and cleaned[0]["scored_text"].startswith(head + " w100")
    p10, p20, p30 = by[10], by[20], by[26]
    assert p10["accepted"] and p10["ai_words"] == 0 and p10["ai_word_share"] == 0.0     # inside the head
    assert p10["ai_tokens"] == 0
    assert not p20["accepted"] and p20["reject_reason"] == "too_short"                # 15 < 19
    assert p30["accepted"] and p30["n_words"] == 25   # 25 >= 0.95 * 26 and p30["ai_words"] == 10 and p30["ai_word_share"] == 0.4
    assert p30["passage"][p30["ai_char_start"]:].startswith("w100")
    assert p30["ai_tokens"] == p30["n_tokens"] - one_token_per_char(head)
    assert "head" not in p30 and p30["k"] == 50


def test_continue_linebreak_separator_and_empty_or_unjoined():
    head = "पहली पंक्ति\nदूसरी पंक्ति"
    passages, cleaned, *_ = process([c_row(head, "तीसरी पंक्ति\nचौथी पंक्ति", "punct+linebreak",
                                           cutoffs={"words": [8]}, language="hi")], Options(mode="gen"), None)
    assert cleaned[0]["scored_text"] == head + "\nतीसरी पंक्ति\nचौथी पंक्ति"
    assert passages[0]["accepted"] and passages[0]["ai_word_share"] == 0.5
    passages, *_ = process([c_row(f"{words(20)}.", "  ")], Options(mode="gen"), None)
    assert all(p["reject_reason"] == "empty" for p in passages)
    passages, *_ = process([c_row(f"{words(20)}.", f"{words(10)}.")], Options(mode="gen", join_head=False), None)
    assert passages[0]["accepted"] and "ai_char_start" not in passages[0] and passages[0]["n_words"] == 10
    with pytest.raises(ValueError):
        process([{**c_row("h.", "c."), "head": None}], Options(mode="gen"), None)


# --------------------------------------------------------------------------- slack floor


def test_slack_floor_and_default_rule():
    text = f"{words(85)}. {words(30, 85)}."
    assert trim.min_units(100) == 95 and trim.min_units(100, 0.95, 15) == 85 and trim.min_units(100, 0.95, 2) == 95
    assert not trim_text(text, "words", 100, None).accepted
    r = trim_text(text, "words", 100, None, 0.95, SENT_END_DEFAULT, 15)
    assert r.accepted and r.min_units == 85
    # positional call as build_edit_prompts makes it still works (6 args)
    assert trim_text(text, "words", 100, None, 0.95, SENT_END_DEFAULT).min_units == 95
    passages, *_ = process([{"id": "h", "language": "en", "text": text}],
                           Options(mode="human", words=[100], slack={"words": 15, "tokens": 0}), None)
    assert passages[0]["accepted"] and passages[0]["slack"] == 15 and passages[0]["min_units"] == 85


def test_main_records_rule_and_uses_promptset_defaults(tmp_path):
    src = tmp_path / "h.jsonl"
    text = " ".join(f"{words(19, i * 19)}." for i in range(80))  # 19-word sentences, 1520 words
    src.write_text(json.dumps({"id": "h", "language": "en", "text": text}) + "\n")
    out = tmp_path / "o.jsonl"
    assert trim.main(["human", "--in", str(src), "--out", str(out), "--tokenizer", "none", "--slack-words", "20"]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert sorted(r["cutoff"] for r in rows) == [100, 200, 400, 800, 1200]   # promptset.yaml cutoffs.words
    assert all(r["accepted"] for r in rows)                                    # 95 >= min(95, 80), ...
    meta = json.loads((tmp_path / "o.summary.json").read_text())["meta"]
    assert meta["min_frac"] == 0.95 and meta["slack"] == {"words": 20, "tokens": 0}
    assert "min(0.95 * N, N - slack)" in meta["acceptance_rule"]
    assert "Acceptance rule" in (tmp_path / "o.summary.md").read_text()
