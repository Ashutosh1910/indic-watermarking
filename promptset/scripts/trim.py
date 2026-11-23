"""Trim AI generations and human passages to fixed lengths with one rule (step after generate.py).

Rule: cut each text at the LAST SENTENCE END AT OR BEFORE cutoff N, and accept the passage only if it
still has at least ``min(min_frac * N, N - slack)`` units ("95 % or one sentence of slack", see
"Acceptance threshold" below); otherwise reject it (and, for generations, list the id for
regeneration). Two kinds of cutoff:

* ``words``  -- whitespace-split words (the writer's view);
* ``tokens`` -- Qwen3 tokens of the re-encoded passage, ``tokenizer(passage, add_special_tokens=False)``
  (the detector's view: exactly what ``SynthIDVerifier.verify_text`` scores).

Sentence end: one or more of ``. ! ? । ॥`` or ``…`` (so "..." and "?!" count), optionally followed by
closing quotes/brackets (``" ' ” ’ ) ] »`` in any combination), followed by whitespace or the end of
the text. A paragraph break after text WITHOUT final punctuation is NOT a sentence end (so an
unpunctuated heading or a cut-off line never ends a passage). Abbreviations ("Dr.", "e.g.") do count
as ends; this only matters if one sits right at the cutoff. ``--ascii-danda`` also
treats "|" as a danda (some web Hindi uses it).

Per-row ``sentence_end_rule`` (both modes): "punct" (default, the rule above) or "punct+linebreak" for
text whose punctuation was stripped upstream (hindi-aesthetics: one sentence per line), where every line
break, and the end of the text, also counts as a sentence end. For those rows the cleaning step's
"short unpunctuated first line = title" heuristic is off.

Acceptance threshold (``min_units`` and ``slack`` in every row): a passage of n units cut at N is accepted
iff n >= min(min_frac * N, N - slack). Defaults come from promptset.yaml: min_frac 0.95 and slack_words 30,
i.e. one sentence of slack:

* word cutoffs:  slack = slack_words (30 words);
* token cutoffs: slack = 30 words' worth of tokens in the row's language,
  round(slack_words * tokens_per_word[lang]) -> en 38, hi 137 (a language without tokens_per_word gets 0).

The 95 % rule still decides long cutoffs (min picks it once 0.05 * N > slack: N > 600 words, N > 760 en /
2740 hi tokens). Overrides: ``--min-frac F``; ``--slack-words S`` replaces 30 for word cutoffs and for the
derived token slack; ``--slack-tokens T`` applies T tokens to every language. ``--slack-words 0`` (without
``--slack-tokens``) turns the slack off: the pure rule n >= min_frac * N. The rule used, with the slack per
kind and language, is written into the summary. (The Python API -- ``trim_text``, ``min_units``,
``Options`` -- defaults to slack 0; the CLI and build_edit_prompts.py apply promptset.yaml's slack.)

C_continue rows (gen mode): generate.py's "text" is only the continuation. The scored text is the row's
``head`` + separator + cleaned continuation (separator: a newline for "punct+linebreak" rows, else a
space, as build_edit_prompts.py defines). Each passage records ``ai_char_start`` (offset in the passage
where the AI text starts), ``ai_words`` / ``ai_word_share`` and ``ai_tokens`` / ``ai_token_share``;
small cutoffs can fall entirely inside the human head (share 0). ``--no-join-head`` trims the
continuation alone. An empty continuation is rejected as "empty".

Modes (same code path; only gen mode cleans the text first):

    # generations: cutoffs from each row's "cutoffs" field (override with --words / --tokens)
    python trim.py gen   --in runs/hi.s*.jsonl --out trimmed/hi.jsonl --tokenizer Qwen/Qwen3-8B
    # human passages (rows: id, language, text, + metadata): cutoffs from --words / --tokens
    python trim.py human --in human_hi.jsonl --out trimmed/human_hi.jsonl --words 200 400 --tokens 200 400

Gen mode with several attempts per id (regenerations): for each cutoff the FIRST attempt (lowest
number) that is accepted is used; if none is, the last attempt's rejected passage is reported.

Cleaning before trimming (gen mode; each step can be switched off with --skip-clean STEP; the steps
that changed a text are listed in "cleaning", what they removed in "cleaning_removed"):

    strip           leading/trailing whitespace
    preamble        a first line like "Sure, here is the story:", "Certainly!", "ज़रूर! यह रही कहानी:",
                    "प्रस्तुत है ...", "यहाँ ... है:" (at most 2 such lines)
    title           a first line that is a markdown heading "# ...", a whole-line "**...**", "Title: ...",
                    or a short line (<= 12 words) without sentence-final punctuation followed by a blank line
    trailing_notes  everything from a line starting "Note:" / "Word count" (Hindi: नोट:, टिप्पणी:, शब्द संख्या),
                    or from a "---" / "***" / "___" line when at most 100 words follow it
    markdown        the bold/italic markers "**" and "__"

Outputs (``--out X.jsonl``):

    X.jsonl               one row per (id, cutoff kind, N): id, pair_id, language, level, genre, source,
                          cutoff_kind, cutoff, accepted, reject_reason, passage, n_words, n_tokens, attempt,
                          script_share, script_ok, cleaning, ... + passthrough metadata (heavy fields
                          like messages/token_ids/text are dropped, see --drop-fields)
    X.cleaned.jsonl       one row per (id, attempt): text_raw (as generated), text_clean, cleaning steps
    X.summary.md / .json  acceptance by language x level x cutoff_kind x cutoff, with reject reasons
    X.regen_ids.txt       ids whose LARGEST cutoff (of any kind) failed for length (too_short /
                          no_sentence_end / empty): feed to ``generate.py --only-ids ... --attempt k+1``;
                          X.regen_ids.attempt<k>.txt splits them by the attempt number to use next

Script check: share of letters (Unicode categories L* and M*, so Devanagari vowel signs count) that
are Devanagari (hi) or Latin (en), on the passage. ``script_ok`` is share >= 0.8 for hi and >= 0.9
for en (``--script-min hi=0.8 --script-min en=0.9``); a failing passage is rejected with "script".
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import log, md_table, read_jsonl_files, sibling, write_jsonl  # noqa: E402

KINDS = ("words", "tokens")
LENGTH_REASONS = ("too_short", "no_sentence_end", "empty")
CLEAN_STEPS = ("strip", "preamble", "title", "trailing_notes", "markdown")
# heavy fields not copied into passage rows (C rows' "head" is inside the passage; "human_passage" is the
# human anchor, scored from human_passages.jsonl in human mode)
DEFAULT_DROP_FIELDS = ("messages", "token_ids", "text", "text_raw", "cutoffs", "head", "human_passage")
PROMPTSET_CONFIG = Path(__file__).resolve().parent.parent / "config" / "promptset.yaml"
DEFAULT_SCRIPT_MIN = {"hi": 0.8, "en": 0.9}
SCRIPT_NAME = {"hi": "DEVANAGARI", "en": "LATIN"}

# --------------------------------------------------------------------------- sentence ends

_CLOSERS = "\"'”’)\\]»"
_TERMINALS = ".!?।॥…"


def _sent_end_re(ascii_danda: bool = False) -> re.Pattern[str]:
    term = _TERMINALS + ("|" if ascii_danda else "")
    return re.compile(f"[{re.escape(term)}]+[{_CLOSERS}]*(?=\\s|$)")


SENT_END = _sent_end_re(False)

# "punct+linebreak" rule, for text whose punctuation was removed upstream (one sentence per line, e.g. the
# hindi-aesthetics corpus): every line break is ALSO a sentence end. The match ends right after the last
# non-space character of each line (and of the text). Same regex as build_edit_prompts.LINE_END.
SENTENCE_END_RULES = ("punct", "punct+linebreak")


def sentence_end_pattern(rule: str = "punct", ascii_danda: bool = False) -> re.Pattern[str]:
    """Pattern for a row's ``sentence_end_rule``: "punct" (default) or "punct+linebreak"."""
    base = _sent_end_re(ascii_danda)
    if rule == "punct":
        return base
    if rule == "punct+linebreak":
        return re.compile(base.pattern + r"|\S(?=[ \t]*(?:\r?\n|$))")
    raise ValueError(f"unknown sentence_end_rule {rule!r} (expected one of {SENTENCE_END_RULES})")


LINE_END = sentence_end_pattern("punct+linebreak")


def sentence_ends(text: str, pattern: re.Pattern[str] = SENT_END) -> list[int]:
    """Offsets just after each sentence end (punctuation + closing quotes), ascending."""
    return [m.end() for m in pattern.finditer(text)]


def count_words(text: str) -> int:
    return len(text.split())


# --------------------------------------------------------------------------- trimming


@dataclass
class Trim:
    passage: str
    n_words: int
    n_tokens: int | None
    accepted: bool
    reject_reason: str | None
    full_units: int | None  # size of the whole text in the cutoff's unit
    min_units: float | None = None  # acceptance threshold used: min(min_frac * N, N - slack)


def min_units(cutoff: int, min_frac: float = 0.95, slack: int = 0) -> float:
    """Acceptance threshold: a passage of n units is accepted iff n >= min(min_frac * N, N - slack).
    With slack 0 this is the pure fraction rule n >= min_frac * N."""
    return min(min_frac * cutoff, cutoff - slack) if slack > 0 else min_frac * cutoff


# Slack spec: {"words": int, "tokens": int | {lang: int}} (an int applies to every language).
SlackSpec = dict[str, Any]
DEFAULT_SLACK_WORDS = 30   # fallback when promptset.yaml has no slack_words


def _round_half_up(x: float) -> int:
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def resolve_slack(slack_words: int, tokens_per_word: dict[str, float] | None = None,
                  slack_tokens: int | None = None) -> SlackSpec:
    """The slack per kind (and, for tokens, per language) of the rule n >= min(min_frac * N, N - slack).

    Word slack = ``slack_words``. Token slack = ``slack_tokens`` for every language if given, else
    ``round(slack_words * tokens_per_word[lang])`` per language (30 words -> en 38, hi 137 tokens)."""
    sw = int(slack_words)
    if sw < 0 or (slack_tokens is not None and int(slack_tokens) < 0):
        raise ValueError("slack must be >= 0")
    if slack_tokens is not None:
        return {"words": sw, "tokens": int(slack_tokens)}
    return {"words": sw, "tokens": {str(lang): _round_half_up(sw * float(t))
                                    for lang, t in sorted((tokens_per_word or {}).items())}}


def slack_for(slack: SlackSpec | None, kind: str, language: str | None) -> int:
    """Slack (in the cutoff's unit) for one kind and language; 0 when not configured."""
    v = (slack or {}).get(kind, 0)
    if isinstance(v, dict):
        v = v.get(str(language), 0)
    return int(v or 0)


def describe_rule(min_frac: float, slack: SlackSpec | None, slack_words: int | None = None) -> str:
    """One line stating the acceptance rule, e.g. for summaries and build metadata."""
    sw = slack_for(slack, "words", None)
    tok = (slack or {}).get("tokens", 0)
    tok_any = any(tok.values()) if isinstance(tok, dict) else bool(tok)
    if not sw and not tok_any:
        return f"accept iff n >= {min_frac} * N (no slack)"
    if isinstance(tok, dict):
        per = ", ".join(f"{lang} {v}" for lang, v in tok.items()) or "none configured"
        src = f" = round({slack_words} x tokens_per_word)" if slack_words is not None else ""
        tok_txt = f"tokens {per}{src}; other languages 0"
    else:
        tok_txt = f"tokens {tok} (every language)"
    return (f"accept iff n >= min({min_frac} * N, N - slack), N = cutoff, n = units of the passage cut at the "
            f"last sentence end at or before N; slack: words {sw}; {tok_txt}")


Counter_ = Callable[[str], int]


def trim_text(text: str, kind: str, cutoff: int, count_tokens: Counter_ | None, min_frac: float = 0.95,
              pattern: re.Pattern[str] = SENT_END, slack: int = 0) -> Trim:
    """Cut ``text`` at its last sentence end whose prefix has <= ``cutoff`` units (words or tokens).

    Accept when the passage has >= ``min(min_frac * cutoff, cutoff - slack)`` units (slack 0: the 95 %
    rule). ``pattern`` is the sentence-end regex (see :func:`sentence_end_pattern`). Token counts assume that a longer prefix
    never has fewer tokens than a shorter one (true for BPE up to rare merges at the boundary), so the
    largest qualifying end is found by binary search; every reported count is exact.
    """
    if kind not in KINDS:
        raise ValueError(f"cutoff kind must be one of {KINDS}")
    if kind == "tokens" and count_tokens is None:
        raise ValueError("token cutoffs need a tokenizer")
    def ntok(s: str) -> int | None:
        return count_tokens(s) if count_tokens else None

    thr = min_units(cutoff, min_frac, slack)
    if not text.strip():
        return Trim("", 0, 0 if count_tokens else None, False, "empty", 0, thr)
    ends = sentence_ends(text, pattern)

    if kind == "words":
        word_starts = [m.start() for m in re.finditer(r"\S+", text)]
        full_units = len(word_starts)
        # words in text[:e] = words starting before e (an end is always at a word boundary)
        best = None
        for e in ends:
            if bisect.bisect_left(word_starts, e) <= cutoff:
                best = e
            else:
                break
    else:
        full_units = count_tokens(text.strip())  # type: ignore[misc]
        lo, hi, best = 0, len(ends) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if count_tokens(text[: ends[mid]].strip()) <= cutoff:  # type: ignore[misc]
                best, lo = ends[mid], mid + 1
            else:
                hi = mid - 1

    if best is None:
        return Trim("", 0, 0 if count_tokens else None, False, "no_sentence_end", full_units, thr)
    passage = text[:best].strip()
    nw, nt = count_words(passage), ntok(passage)
    units = nw if kind == "words" else nt
    if units < thr:
        return Trim(passage, nw, nt, False, "too_short", full_units, thr)
    return Trim(passage, nw, nt, True, None, full_units, thr)


# --------------------------------------------------------------------------- script check


def script_share(text: str, language: str) -> float | None:
    """Share of letters (categories L*, M*) whose Unicode name starts with the language's script name."""
    script = SCRIPT_NAME.get(language)
    if script is None:
        return None
    letters = target = 0
    for ch in text:
        if unicodedata.category(ch)[0] in "LM":
            letters += 1
            if unicodedata.name(ch, "").startswith(script):
                target += 1
    return target / letters if letters else 0.0


# --------------------------------------------------------------------------- cleaning


def _nfd(s: str) -> str:
    return unicodedata.normalize("NFD", s)


_EN_START = (r"sure|certainly|of course|absolutely|okay|ok|great|alright|all right|here(?:'s|’s| is| are)|"
             r"below is|below are|as requested|i(?:'d|’d| would) be (?:happy|glad)")
_HI_START = (r"ज़रूर|जरूर|निश्चित रूप से|बिल्कुल|बिलकुल|अवश्य|यह रही|यह रहा|यह रहे|ये रही|ये रहा|ये रहे|यह है|"
             r"प्रस्तुत है|प्रस्तुत हैं|यहाँ|यहां|नीचे")
_PREAMBLE_START = re.compile(_nfd(rf"^\W*(?:{_EN_START}|{_HI_START})(?=\W|$)"), re.I)
_INTERJECTION = re.compile(_nfd(r"^\W*(?:sure|certainly|of course|absolutely|okay|ok|great|alright|"
                                r"ज़रूर|जरूर|बिल्कुल|बिलकुल|अवश्य|निश्चित रूप से)[\s!.,।:]*$"), re.I)
_META_WORDS = re.compile(_nfd(
    r"\b(?:story|essay|poem|text|version|rewrite|rewritten|paraphrase|paraphrased|continuation|outline|"
    r"passage|draft|response|revised|polished|proofread|corrected|edited|article|piece|summary)\b|"
    r"कहानी|निबंध|कविता|पाठ|संस्करण|रूपरेखा|अनुच्छेद|लेख|संशोधित|जारी|उत्तर|पैराग्राफ|सारांश|प्रस्तुत"), re.I)
_TITLE_MD = re.compile(r"^\s*(?:#{1,6}\s+\S.*|\*\*[^*].*\*\*:?|__[^_].*__:?)\s*$")
_TITLE_LABEL = re.compile(_nfd(r"^\W*(?:title|शीर्षक)\s*[:：]"), re.I)
_FINAL_PUNCT = re.compile(f"[{re.escape(_TERMINALS)}][{_CLOSERS}*_]*\\s*$")
_NOTE_LINE = re.compile(_nfd(r"^[\s*_(\[]*(?:(?:author'?s |editor'?s )?notes?\s*[:：]|word[ -]count\b|"
                             r"नोट\s*[:：]|टिप्पणी\s*[:：]|शब्द[ -]?संख्या)"), re.I)
_SEPARATOR = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
TITLE_MAX_WORDS = 12
PREAMBLE_MAX_WORDS = 30
SEPARATOR_TAIL_MAX_WORDS = 100


@dataclass
class Cleaned:
    text: str
    steps: list[str] = field(default_factory=list)
    removed: dict[str, str] = field(default_factory=dict)

    def note(self, step: str, removed: str) -> None:
        if step not in self.steps:
            self.steps.append(step)
        if not removed:
            return
        prev = self.removed.get(step)
        self.removed[step] = removed if prev is None else prev + "\n" + removed


def _split_first_line(text: str) -> tuple[str, str]:
    first, sep, rest = text.partition("\n")
    return first, rest if sep else ""


def is_preamble_line(line: str) -> bool:
    s = _nfd(line.strip())
    if not s or count_words(s) > PREAMBLE_MAX_WORDS:
        return False
    if _INTERJECTION.match(s):
        return True
    if not _PREAMBLE_START.match(s):
        return False
    return s.rstrip("*_ ").endswith((":", "：")) or bool(_META_WORDS.search(s))


def is_title_line(line: str, rest: str, short_line_titles: bool = True) -> bool:
    s = line.strip()
    if not s:
        return False
    if _TITLE_MD.match(s) or _TITLE_LABEL.match(_nfd(s)):
        return True
    if not short_line_titles:
        return False
    followed_by_blank = re.match(r"[ \t]*\n", rest) is not None
    return count_words(s) <= TITLE_MAX_WORDS and not _FINAL_PUNCT.search(s) and followed_by_blank


def clean_text(text: str, steps: Iterable[str] = CLEAN_STEPS, short_line_titles: bool = True) -> Cleaned:
    """Apply the enabled cleaning steps in a fixed order: strip, preamble, title, trailing_notes, markdown.

    ``short_line_titles=False`` keeps the "short unpunctuated first line + blank line" title heuristic off
    (used for "punct+linebreak" rows, where no line has final punctuation); markdown/label titles still go.
    """
    on = set(steps)
    bad = on - set(CLEAN_STEPS)
    if bad:
        raise ValueError(f"unknown cleaning step(s) {sorted(bad)}")
    c = Cleaned(text)

    def strip() -> None:
        if "strip" in on and c.text != c.text.strip():
            c.note("strip", "")
            c.text = c.text.strip()

    strip()
    if "preamble" in on:
        for _ in range(2):
            first, rest = _split_first_line(c.text.lstrip())
            if rest.strip() and is_preamble_line(first):
                c.note("preamble", first.strip())
                c.text = rest.lstrip("\n") if "strip" not in on else rest.strip()
            else:
                break
    if "title" in on:
        first, rest = _split_first_line(c.text.lstrip())
        if rest.strip() and is_title_line(first, rest, short_line_titles):
            c.note("title", first.strip())
            c.text = rest.lstrip("\n") if "strip" not in on else rest.strip()
    if "trailing_notes" in on:
        lines = c.text.split("\n")
        for i in range(1, len(lines)):
            tail = "\n".join(lines[i:])
            if _NOTE_LINE.match(_nfd(lines[i])) or (
                    _SEPARATOR.match(lines[i]) and count_words(tail) - 1 <= SEPARATOR_TAIL_MAX_WORDS):
                head = "\n".join(lines[:i])
                if head.strip():
                    c.note("trailing_notes", tail.strip())
                    c.text = head
                break
    if "markdown" in on and ("**" in c.text or "__" in c.text):
        n = c.text.count("**") + c.text.count("__")
        c.note("markdown", f"{n} markers")
        c.text = c.text.replace("**", "").replace("__", "")
    strip()
    return c


# --------------------------------------------------------------------------- rows


@dataclass
class Options:
    mode: str
    min_frac: float = 0.95
    words: list[int] | None = None
    tokens: list[int] | None = None
    default_cutoffs: dict[str, list[int]] | None = None   # used when neither CLI nor row has cutoffs
    # {"words": int, "tokens": int | {lang: int}}; the CLI fills it from promptset.yaml (resolve_slack)
    slack: SlackSpec = field(default_factory=lambda: {"words": 0, "tokens": 0})
    slack_words: int | None = None  # the slack_words the token slack was derived from (for rule_text)
    clean_steps: tuple[str, ...] = CLEAN_STEPS
    script_min: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCRIPT_MIN))
    drop_fields: tuple[str, ...] = DEFAULT_DROP_FIELDS
    regen_reasons: tuple[str, ...] = LENGTH_REASONS
    ascii_danda: bool = False
    join_head: bool = True          # gen mode: C_continue rows are scored as head + separator + continuation
    default_rule: str = "punct"     # sentence_end_rule for rows without one
    skip_kinds: tuple[str, ...] = ()  # cutoff kinds ignored when they come from rows/defaults

    def pattern_for(self, row: dict[str, Any]) -> re.Pattern[str]:
        return sentence_end_pattern(rule_of(row, self.default_rule), self.ascii_danda)

    def slack_for(self, kind: str, language: str | None) -> int:
        return slack_for(self.slack, kind, language)

    def rule_text(self) -> str:
        return describe_rule(self.min_frac, self.slack, self.slack_words)


def rule_of(row: dict[str, Any], default: str = "punct") -> str:
    rule = row.get("sentence_end_rule") or default
    if rule not in SENTENCE_END_RULES:
        raise ValueError(f"row {row.get('id')}: unknown sentence_end_rule {rule!r}")
    return rule


def row_cutoffs(row: dict[str, Any], opt: Options) -> dict[str, list[int]]:
    """Cutoffs per kind: --words/--tokens if either is given (only those), else the row's "cutoffs" field,
    else the defaults (promptset.yaml's cutoffs, human mode). ``opt.skip_kinds`` drops a kind (token
    cutoffs from rows/defaults when there is no tokenizer)."""
    cli_given = opt.words is not None or opt.tokens is not None
    src = row.get("cutoffs") or {}
    out: dict[str, list[int]] = {}
    for kind in KINDS:
        if cli_given:
            vals = getattr(opt, kind)
        elif kind in opt.skip_kinds:
            vals = None
        elif src:
            vals = src.get(kind)
        else:
            vals = (opt.default_cutoffs or {}).get(kind)
        if vals:
            out[kind] = sorted({int(v) for v in vals})
    return out


# --------------------------------------------------------------------------- C_continue: head + continuation


CONTINUE_LEVEL = "C_continue"


def is_continuation(row: dict[str, Any]) -> bool:
    return str(row.get("level", "")).startswith(CONTINUE_LEVEL)


def join_separator(row: dict[str, Any]) -> str:
    """How build_edit_prompts.py says head and continuation join: a newline for punct+linebreak rows, else a space."""
    return "\n" if rule_of(row) == "punct+linebreak" else " "


@dataclass
class Scored:
    """The text that is trimmed and scored, and where the AI-written part starts in it."""
    text: str
    ai_char_start: int | None = None   # None: the whole text is AI (or human, in human mode)
    ai_text_empty: bool = False


def scored_text(row: dict[str, Any], cleaned_ai: str, opt: Options) -> Scored:
    if not (opt.mode == "gen" and opt.join_head and is_continuation(row)):
        return Scored(cleaned_ai)
    head = row.get("head")
    if head is None:
        raise ValueError(f"row {row.get('id')}: C_continue row without a 'head' field "
                         "(use --no-join-head to trim the continuation alone)")
    head = head.strip()
    ai = cleaned_ai.strip()
    sep = join_separator(row)
    return Scored(head + sep + ai, len(head) + len(sep), not ai)


def ai_share(passage: str, ai_char_start: int | None, n_words: int, n_tokens: int | None,
             count_tokens: Counter_ | None) -> dict[str, Any]:
    """AI part of a trimmed C passage: words/tokens after ``ai_char_start`` and their share.

    Head and continuation are joined at whitespace, so the word split is exact; the token split counts the
    head alone (``n_tokens - tokens(head)``), exact up to a merge across the separator (Qwen3 pre-tokenises
    on whitespace, so this does not happen for a space or newline)."""
    if ai_char_start is None:
        return {}
    human = passage[:ai_char_start].strip()
    ai = passage[ai_char_start:] if len(passage) > ai_char_start else ""
    ai_words = count_words(ai)
    out: dict[str, Any] = {"ai_char_start": min(ai_char_start, len(passage)), "ai_words": ai_words,
                           "ai_word_share": round(ai_words / n_words, 4) if n_words else None}
    if count_tokens is not None and n_tokens is not None:
        ai_tokens = n_tokens - count_tokens(human) if ai.strip() else 0
        out.update(ai_tokens=ai_tokens, ai_token_share=round(ai_tokens / n_tokens, 4) if n_tokens else None)
    return out


# --------------------------------------------------------------------------- rows


def evaluate(text: str, language: str, kind: str, cutoff: int, count_tokens: Counter_ | None,
             opt: Options, pattern: re.Pattern[str] = SENT_END, ai_text_empty: bool = False
             ) -> tuple[Trim, float | None, bool | None]:
    """Trim one text for one cutoff and apply the script check (length reasons take precedence).
    An empty AI part (C rows) rejects the row as "empty" whatever the head gives."""
    t = trim_text(text, kind, cutoff, count_tokens, opt.min_frac, pattern, opt.slack_for(kind, language))
    if ai_text_empty:
        return Trim(t.passage, t.n_words, t.n_tokens, False, "empty", t.full_units, t.min_units), None, None
    share = script_share(t.passage if t.passage else text, language)
    thr = opt.script_min.get(language)
    ok = None if share is None or thr is None else share >= thr
    if t.accepted and ok is False:
        t = Trim(t.passage, t.n_words, t.n_tokens, False, "script", t.full_units, t.min_units)
    return t, share, ok


def _row_key(row: dict[str, Any]) -> tuple[str, bool]:
    return str(row["id"]), bool(row.get("watermarked", False))


def process(rows: Sequence[dict[str, Any]], opt: Options, count_tokens: Counter_ | None
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, bool], int], list[str]]:
    """Returns (passage rows, cleaned rows, next attempt per regen id, ids without cutoffs)."""
    groups: dict[tuple[str, bool], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[_row_key(r)].append(r)
    passages: list[dict[str, Any]] = []
    cleaned_rows: list[dict[str, Any]] = []
    regen: dict[tuple[str, bool], int] = {}
    no_cutoffs: list[str] = []

    for key, attempts in groups.items():
        attempts.sort(key=lambda r: int(r.get("attempt") or 0))
        seen_attempts: set[int] = set()
        cleaned: list[tuple[dict[str, Any], Cleaned, Scored]] = []
        for r in attempts:
            a = int(r.get("attempt") or 0)
            if a in seen_attempts:
                log(f"warning: id {key[0]} attempt {a} appears twice; keeping the first")
                continue
            seen_attempts.add(a)
            raw = r.get("text") or ""
            if opt.mode == "gen":
                c = clean_text(raw, opt.clean_steps, short_line_titles=rule_of(r, opt.default_rule) == "punct")
            else:
                c = Cleaned(raw)
            sc = scored_text(r, c.text, opt)
            cleaned.append((r, c, sc))
            if opt.mode == "gen":
                rec = {
                    "id": key[0], "attempt": a, "watermarked": r.get("watermarked"),
                    "language": r.get("language"), "level": r.get("level"),
                    "finish_reason": r.get("finish_reason"), "cleaning": c.steps, "cleaning_removed": c.removed,
                    "text_raw": raw, "text_clean": c.text,
                }
                if sc.ai_char_start is not None:
                    rec.update(scored_text=sc.text, ai_char_start=sc.ai_char_start)
                cleaned_rows.append(rec)
        cut = row_cutoffs(attempts[0], opt)
        if not cut:
            no_cutoffs.append(key[0])
            continue
        pattern = opt.pattern_for(attempts[0])
        tried = [int(r.get("attempt") or 0) for r, _, _ in cleaned]
        for kind, values in cut.items():
            for n in values:
                chosen = None
                for r, c, sc in cleaned:
                    res = evaluate(sc.text, r.get("language", ""), kind, n, count_tokens, opt, pattern,
                                   sc.ai_text_empty)
                    chosen = (r, c, sc, res)
                    if res[0].accepted:
                        break
                assert chosen is not None
                r, c, sc, (t, share, ok) = chosen
                extra = ai_share(t.passage, sc.ai_char_start, t.n_words, t.n_tokens, count_tokens)
                passages.append(passage_row(r, c, kind, n, t, share, ok, tried, opt, extra))
                if n == values[-1] and not t.accepted and t.reject_reason in opt.regen_reasons:
                    regen[key] = max(tried) + 1 if opt.mode == "gen" else 0
    return passages, cleaned_rows, regen, no_cutoffs


def passage_row(r: dict[str, Any], c: Cleaned, kind: str, n: int, t: Trim, share: float | None,
                ok: bool | None, tried: list[int], opt: Options, extra: dict[str, Any] | None = None
                ) -> dict[str, Any]:
    head = {
        "id": str(r["id"]), "pair_id": r.get("pair_id"), "language": r.get("language"),
        "level": r.get("level", "human" if opt.mode == "human" else None), "genre": r.get("genre"),
        "source": r.get("source"), "k": r.get("k"), "cutoff_kind": kind, "cutoff": n, "accepted": t.accepted,
        "reject_reason": t.reject_reason, "n_words": t.n_words, "n_tokens": t.n_tokens,
        "full_units": t.full_units, "min_units": t.min_units,
        "attempt": r.get("attempt") if opt.mode == "gen" else None,
        "attempts_tried": tried if opt.mode == "gen" else None, "watermarked": r.get("watermarked"),
        "script_share": None if share is None else round(share, 4), "script_ok": ok,
        "cleaning": c.steps, "trim_mode": opt.mode, "min_frac": opt.min_frac,
        "slack": opt.slack_for(kind, r.get("language")), "sentence_end_rule": rule_of(r, opt.default_rule),
        **(extra or {}),
        "passage": t.passage,
    }
    rest = {k: v for k, v in r.items() if k not in head and k not in opt.drop_fields}
    return {**head, **rest}


# --------------------------------------------------------------------------- summary


GROUP_KEYS = ("language", "level", "k", "cutoff_kind", "cutoff")


def _mean(values: Iterable[float | None]) -> float | None:
    vs = [v for v in values if v is not None]
    return sum(vs) / len(vs) if vs else None


def summarize(passages: Sequence[dict[str, Any]], group_keys: Sequence[str] = GROUP_KEYS) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for p in passages:
        groups[tuple(p.get(k) for k in group_keys)].append(p)
    out = []
    def order(k: tuple) -> tuple:
        return tuple((0, v, "") if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, 0, str(v))
                     for v in k)

    for key in sorted(groups, key=order):
        ps = groups[key]
        acc = [p for p in ps if p["accepted"]]
        reasons = Counter(p["reject_reason"] for p in ps if not p["accepted"])
        att = Counter(p.get("attempt") for p in acc)
        out.append({
            **dict(zip(group_keys, key)), "n": len(ps), "accepted": len(acc),
            "rate": len(acc) / len(ps) if ps else 0.0, "reject_reasons": dict(reasons),
            "accepted_by_attempt": {str(k): v for k, v in sorted(att.items(), key=lambda kv: str(kv[0]))},
            "mean_words_accepted": sum(p["n_words"] for p in acc) / len(acc) if acc else None,
            "mean_tokens_accepted": (sum(p["n_tokens"] for p in acc) / len(acc)
                                     if acc and acc[0]["n_tokens"] is not None else None),
            "mean_ai_word_share_accepted": _mean(p.get("ai_word_share") for p in acc),
            "mean_ai_token_share_accepted": _mean(p.get("ai_token_share") for p in acc),
        })
    return out


def summary_markdown(summary: Sequence[dict[str, Any]], group_keys: Sequence[str], title: str) -> str:
    reasons = sorted({r for s in summary for r in s["reject_reasons"]})
    has_ai = any(s["mean_ai_word_share_accepted"] is not None for s in summary)
    header = (list(group_keys) + ["n", "accepted", "rate"] + reasons + ["mean words", "mean tokens"]
              + (["AI word share", "AI token share"] if has_ai else []))
    rows = [[s.get(k) for k in group_keys] + [s["n"], s["accepted"], s["rate"]]
            + [s["reject_reasons"].get(r, 0) for r in reasons]
            + [None if s["mean_words_accepted"] is None else round(s["mean_words_accepted"], 1),
               None if s["mean_tokens_accepted"] is None else round(s["mean_tokens_accepted"], 1)]
            + ([s["mean_ai_word_share_accepted"], s["mean_ai_token_share_accepted"]] if has_ai else [])
            for s in summary]
    return f"# {title}\n\n" + md_table(header, rows) + "\n"


# --------------------------------------------------------------------------- CLI


def load_promptset_config(path: str | None) -> dict[str, Any]:
    """min_frac, slack_words, tokens_per_word and cutoffs from promptset/config/promptset.yaml
    ({} if missing or unreadable)."""
    if not path or not Path(path).exists():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover
        return {}
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return {k: raw[k] for k in ("min_frac", "slack_words", "tokens_per_word", "cutoffs") if k in raw}


def make_token_counter(name: str | None) -> Counter_ | None:
    if not name or name.lower() == "none":
        return None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])


def _script_min(specs: Sequence[str] | None) -> dict[str, float]:
    out = dict(DEFAULT_SCRIPT_MIN)
    for s in specs or []:
        lang, _, v = s.partition("=")
        out[lang.strip()] = float(v)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["gen", "human"])
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="input JSONL file(s)")
    ap.add_argument("--out", required=True, help="passages JSONL; summary/cleaned/regen files go next to it")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-8B",
                    help="HF id or local dir for token counts ('none' = word cutoffs only)")
    ap.add_argument("--words", type=int, nargs="*", help="word cutoffs (override rows' cutoffs.words)")
    ap.add_argument("--tokens", type=int, nargs="*", help="token cutoffs (override rows' cutoffs.tokens)")
    ap.add_argument("--min-frac", type=float, default=None,
                    help="accept if passage >= min_frac * N (default: promptset.yaml min_frac, else 0.95)")
    ap.add_argument("--slack-words", type=int, default=None,
                    help="accept if n >= min(min_frac*N, N - slack): word slack, also the base of the derived "
                         "token slack (default: promptset.yaml slack_words, 30; 0 = off, the pure min_frac rule)")
    ap.add_argument("--slack-tokens", type=int, default=None,
                    help="token slack for every language (default: round(slack_words * tokens_per_word[lang]), "
                         "en 38 / hi 137)")
    ap.add_argument("--promptset-config", default=str(PROMPTSET_CONFIG),
                    help="promptset.yaml: default min_frac, and cutoffs for rows without any (human mode)")
    ap.add_argument("--no-join-head", action="store_true",
                    help="gen mode: trim C_continue continuations alone instead of head + continuation")
    ap.add_argument("--skip-clean", action="append", default=[], choices=CLEAN_STEPS, help="disable a cleaning step")
    ap.add_argument("--script-min", action="append", metavar="LANG=SHARE", help="e.g. hi=0.8 (default hi=0.8, en=0.9)")
    ap.add_argument("--ascii-danda", action="store_true", help='also treat "|" as a sentence end')
    ap.add_argument("--drop-fields", default=",".join(DEFAULT_DROP_FIELDS),
                    help="input fields not copied into passage rows (comma list; text_raw is in X.cleaned.jsonl)")
    ap.add_argument("--regen-reasons", default=",".join(LENGTH_REASONS),
                    help="reject reasons of the largest cutoff that put an id in X.regen_ids.txt")
    ap.add_argument("--group-by", default=",".join(GROUP_KEYS), help="summary grouping keys")
    args = ap.parse_args(argv)

    ps_cfg = load_promptset_config(args.promptset_config)
    min_frac = args.min_frac if args.min_frac is not None else float(ps_cfg.get("min_frac", 0.95))
    slack_words = (args.slack_words if args.slack_words is not None
                   else int(ps_cfg.get("slack_words", DEFAULT_SLACK_WORDS)))
    tpw = ps_cfg.get("tokens_per_word") or {}
    slack = resolve_slack(slack_words, tpw, args.slack_tokens)
    opt = Options(
        mode=args.mode, min_frac=min_frac, words=args.words, tokens=args.tokens,
        default_cutoffs=(ps_cfg.get("cutoffs") if args.mode == "human" else None),
        slack=slack, slack_words=None if args.slack_tokens is not None else slack_words,
        clean_steps=tuple(s for s in CLEAN_STEPS if s not in args.skip_clean),
        script_min=_script_min(args.script_min),
        drop_fields=tuple(f for f in args.drop_fields.split(",") if f),
        regen_reasons=tuple(f for f in args.regen_reasons.split(",") if f),
        ascii_danda=args.ascii_danda, join_head=not args.no_join_head,
    )
    rows = read_jsonl_files(args.inputs)
    for i, r in enumerate(rows):
        if "id" not in r or "text" not in r:
            raise SystemExit(f"row {i} lacks 'id' or 'text'")
        rule_of(r)  # validates sentence_end_rule early
    if isinstance(slack["tokens"], dict) and slack_words:
        missing = sorted({str(r.get("language")) for r in rows} - set(slack["tokens"]))
        if missing:
            log(f"warning: no tokens_per_word for language(s) {missing}: their token slack is 0 (pure min_frac rule)")
    count_tokens = make_token_counter(args.tokenizer)
    if count_tokens is None:
        if args.tokens:
            raise SystemExit("--tokens given but --tokenizer none")
        if any(row_cutoffs(r, opt).get("tokens") for r in rows):
            log("warning: --tokenizer none: token cutoffs from rows/defaults are skipped (word cutoffs only)")
            opt.skip_kinds = ("tokens",)

    passages, cleaned, regen, no_cut = process(rows, opt, count_tokens)
    out = Path(args.out)
    write_jsonl(out, passages)
    if args.mode == "gen":
        write_jsonl(sibling(out, ".cleaned.jsonl"), cleaned)
    group_keys = [k for k in args.group_by.split(",") if k]
    summary = summarize(passages, group_keys)
    regen_ids = sorted({k[0] for k in regen})
    next_attempts = Counter(regen.values())
    meta = {"mode": args.mode, "inputs": args.inputs, "tokenizer": args.tokenizer, "min_frac": opt.min_frac,
            "slack": opt.slack,
            "slack_config": {"slack_words": slack_words,
                             "slack_words_source": "cli" if args.slack_words is not None else
                             ("promptset.yaml" if "slack_words" in ps_cfg else "built-in default"),
                             "slack_tokens": ("cli: same for every language" if args.slack_tokens is not None
                                              else "round(slack_words * tokens_per_word[lang])"),
                             "tokens_per_word": tpw},
            "acceptance_rule": opt.rule_text(),
            "sentence_end_rules": dict(Counter(rule_of(r) for r in rows)),
            "join_head": opt.join_head, "promptset_config": args.promptset_config if ps_cfg else None,
            "clean_steps": list(opt.clean_steps), "script_min": opt.script_min, "n_input_rows": len(rows),
            "n_passages": len(passages), "n_accepted": sum(p["accepted"] for p in passages),
            "ids_without_cutoffs": no_cut, "n_regen_ids": len(regen_ids),
            "regen_next_attempt": {str(k): v for k, v in sorted(next_attempts.items())}}
    sibling(out, ".summary.json").write_text(
        json.dumps({"meta": meta, "groups": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    md = summary_markdown(summary, group_keys, f"Trimming summary: {out.name} ({args.mode} mode)")
    md += (f"\nAcceptance rule: {opt.rule_text()} (sentence ends at or before N; sentence_end_rule per row: "
           f"{meta['sentence_end_rules']}).\n"
           f"\n{meta['n_accepted']}/{len(passages)} passages accepted; "
           f"{len(regen_ids)} ids need regeneration; {len(no_cut)} rows without cutoffs.\n")
    sibling(out, ".summary.md").write_text(md, encoding="utf-8")
    sibling(out, ".regen_ids.txt").write_text("".join(i + "\n" for i in regen_ids), encoding="utf-8")
    if args.mode == "gen":
        for old in out.parent.glob(sibling(out, ".regen_ids.attempt*.txt").name):
            old.unlink()
        for k in sorted(next_attempts):
            ids_k = sorted({key[0] for key, a in regen.items() if a == k})
            sibling(out, f".regen_ids.attempt{k}.txt").write_text("".join(i + "\n" for i in ids_k), encoding="utf-8")

    print(md)
    log(f"wrote {out} ({len(passages)} rows), {sibling(out, '.summary.md')}, {sibling(out, '.regen_ids.txt')}")
    if no_cut:
        log(f"warning: {len(no_cut)} rows have no cutoffs (e.g. {no_cut[:3]}); give --words/--tokens to trim them")
    if regen_ids and args.mode == "gen":
        nxt = ", ".join(f"--attempt {k} for {v} ids" for k, v in sorted(next_attempts.items()))
        log(f"regenerate with: generate.py ... --only-ids {sibling(out, '.regen_ids.txt')} ({nxt})")
        if len(next_attempts) > 1:
            log("note: regen ids need different attempt numbers; use the per-attempt lists "
                f"{sibling(out, '.regen_ids.attempt<k>.txt').name} with --attempt <k>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
