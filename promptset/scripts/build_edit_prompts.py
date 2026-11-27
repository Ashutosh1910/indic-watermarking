"""Build the edit-level prompt set (L1-L4, C, OX -> O, G_title) from HUMAN documents.

Every row has the same schema as the G set (build_promptset.py), so one generate.py run handles all of
them. See promptset/EDIT_LEVELS.md for the study design and the GPU run order.

Subcommands::

    # 1) the study build: a genre-matched sample per language from the study corpora (hi: hindisamay +
    #    samalochan, en: medium; sampling.n_docs_per_language = 300, sampling.genre_mix story 0.5 / essay 0.5),
    #    human passages cut and the prompt rows written (OX outline-extraction rows included)
    python promptset/scripts/build_edit_prompts.py build \\
        --tokenizer /mnt/project-files/promptset_stage/qwen3_tokenizer/ --out promptset/data/edit
    #    per-corpus mode (n docs per corpus, round-robin over genres): name the corpora
    python promptset/scripts/build_edit_prompts.py build --corpus samalochan --corpus hindi-aesthetics \\
        --n-docs 150 --out ...
    #    generic JSONL input (one doc per line: id, language, title, text, genre, license[, source])
    python promptset/scripts/build_edit_prompts.py build --jsonl docs_en.jsonl --n-docs 150 --out ...
    #    (add --sampling genre_mix to pool named inputs per language with the genre mix)

    # 2) after generate.py has run the OX rows (unwatermarked): turn the outlines into O_outline rows
    python promptset/scripts/build_edit_prompts.py build-outline --gen runs/ox.jsonl \\
        --out promptset/data/edit/O_prompts.jsonl

Outputs of ``build`` (in --out):
    edit_prompts.jsonl        all rows (L1-L4, C_continue x k, G_title, OX_extract_outline)
    human_passages.jsonl      the cut human passages (id, language, text, cutoffs, ...): input of
                              ``trim.py human`` and of the FPR check, identical to each row's human_passage
    titles_review.tsv         the {title} chosen for every sampled doc (or why none), for a manual check
    edit_prompts_summary.md   counts per level x experiment x source x language, token budget, skips
    build_meta.json           arguments, config, tokenizer, counts

Length rule (shared with trim.py, same functions): a passage is cut at the last sentence end at or before N
words or tokens and accepted only if it keeps n >= min(min_frac * N, N - slack) ("95 % or one sentence of
slack"). Defaults from promptset.yaml: min_frac 0.95, slack_words 30; word slack = 30 words, token slack =
round(30 * tokens_per_word[lang]) (en 38, hi 137). Override in edit_levels.yaml ``lengths`` (min_frac,
slack_words, slack_tokens) or with ``build --slack-words S [--slack-tokens T]``; ``--slack-words 0`` is the
pure min_frac rule. Sentence ends come from trim.py (``SENT_END``); for punctuation-stripped corpora
(hindi-aesthetics) every line break is also a sentence end.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[1]
sys.path.insert(0, str(SCRIPTS_DIR))
from _common import dumps, iter_jsonl, log, md_table, read_jsonl_files, stable_hash, write_jsonl  # noqa: E402

# The sentence-end rule and the trimming function are trim.py's, so human inputs and AI outputs are cut by
# the same rule. (A local copy is kept only as a fallback for a checkout without trim.py.)
try:
    from trim import (_CLOSERS, _TERMINALS, DEFAULT_SCRIPT_MIN, DEFAULT_SLACK_WORDS, SENT_END,  # noqa: E402
                      Trim, count_words, describe_rule, min_units, resolve_slack, sentence_ends, slack_for,
                      trim_text)
    TRIM_SOURCE = "trim.py"
except ImportError:  # pragma: no cover - fallback; keep in sync with trim.py
    TRIM_SOURCE = "local fallback"
    DEFAULT_SCRIPT_MIN = {"hi": 0.8, "en": 0.9}
    _CLOSERS = "\"'”’)\\]»"
    _TERMINALS = ".!?।॥…"
    SENT_END = re.compile(f"[{re.escape(_TERMINALS)}]+[{_CLOSERS}]*(?=\\s|$)")

    DEFAULT_SLACK_WORDS = 30

    @dataclass
    class Trim:  # type: ignore[no-redef]
        passage: str
        n_words: int
        n_tokens: int | None
        accepted: bool
        reject_reason: str | None
        full_units: int | None
        min_units: float | None = None

    def count_words(text: str) -> int:  # type: ignore[no-redef]
        return len(text.split())

    def min_units(cutoff, min_frac=0.95, slack=0):  # type: ignore[no-redef]
        return min(min_frac * cutoff, cutoff - slack) if slack > 0 else min_frac * cutoff

    def resolve_slack(slack_words, tokens_per_word=None, slack_tokens=None):  # type: ignore[no-redef]
        if slack_tokens is not None:
            return {"words": int(slack_words), "tokens": int(slack_tokens)}
        return {"words": int(slack_words),
                "tokens": {str(k): int(int(slack_words) * float(v) + 0.5) for k, v in sorted((tokens_per_word or {}).items())}}

    def slack_for(slack, kind, language):  # type: ignore[no-redef]
        v = (slack or {}).get(kind, 0)
        return int((v.get(str(language), 0) if isinstance(v, dict) else v) or 0)

    def describe_rule(min_frac, slack, slack_words=None):  # type: ignore[no-redef]
        return f"accept iff n >= min({min_frac} * N, N - slack), slack {slack}"

    def sentence_ends(text: str, pattern: re.Pattern[str] = SENT_END) -> list[int]:  # type: ignore[no-redef]
        return [m.end() for m in pattern.finditer(text)]

    def trim_text(text, kind, cutoff, count_tokens, min_frac=0.95, pattern=SENT_END, slack=0):  # type: ignore[no-redef]
        thr = min_units(cutoff, min_frac, slack)
        if not text.strip():
            return Trim("", 0, 0 if count_tokens else None, False, "empty", 0, thr)
        ends = sentence_ends(text, pattern)
        if kind == "words":
            ws = [m.start() for m in re.finditer(r"\S+", text)]
            full, best = len(ws), None
            for e in ends:
                if bisect.bisect_left(ws, e) <= cutoff:
                    best = e
                else:
                    break
        else:
            full = count_tokens(text.strip())
            lo, hi, best = 0, len(ends) - 1, None
            while lo <= hi:
                mid = (lo + hi) // 2
                if count_tokens(text[: ends[mid]].strip()) <= cutoff:
                    best, lo = ends[mid], mid + 1
                else:
                    hi = mid - 1
        if best is None:
            return Trim("", 0, 0 if count_tokens else None, False, "no_sentence_end", full, thr)
        p = text[:best].strip()
        nw, nt = count_words(p), (count_tokens(p) if count_tokens else None)
        units = nw if kind == "words" else nt
        return Trim(p, nw, nt, units >= thr, None if units >= thr else "too_short", full, thr)

CONFIG_DIR = REPO_ROOT / "promptset" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "edit_levels.yaml"
DEFAULT_TOKENIZER = Path("/mnt/project-files/promptset_stage/qwen3_tokenizer")
LEVEL_ORDER = ("L1_proofread", "L2_polish", "L3_paraphrase", "L4_rewrite", "C_continue", "G_title",
               "OX_extract_outline", "O_outline")
EXP_SHORT = {"same_words": "sw", "same_tokens": "st"}
KINDS = ("words", "tokens")
# A row serving token cutoff N must be allowed MNT_CUTOFF_FACTOR x (N - head tokens) new tokens: the same
# factor as validate_promptset.MNT_CUTOFF_FACTOR (a test keeps the two equal).
MNT_CUTOFF_FACTOR = 1.25
# Tokens the Qwen3 chat template adds around system + user (role headers, <|im_end|>, the empty think block);
# about 20, rounded up. Only used to keep prompt + max_new_tokens under lengths.max_context.
CHAT_TEMPLATE_TOKENS = 32

# Every line break is a sentence end (for text whose punctuation was removed upstream): the match ends right
# after the last non-space character of each line.
LINE_END = re.compile(SENT_END.pattern + r"|\S(?=[ \t]*(?:\r?\n|$))")


def sentence_pattern(punctuation_removed: bool) -> re.Pattern[str]:
    return LINE_END if punctuation_removed else SENT_END


# --------------------------------------------------------------------------- small helpers


def seed_of(s: str) -> int:
    """Same per-row seed as the G set (build_promptset.py)."""
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def short_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def ceil_to(x: float, step: int) -> int:
    return int(math.ceil(x / step) * step) if x > 0 else step


def round_to(x: float, step: int) -> int:
    return max(step, int(round(x / step)) * step) if step > 1 else int(round(x))


def cutoff_need(kind: str, n: int, tokens: int, words: int, head_tokens: int = 0) -> float:
    """New tokens a row needs so that (head +) generation can reach cutoff ``n`` of ``kind``, by the rule
    validate_promptset.py checks: MNT_CUTOFF_FACTOR x (n - head) for token cutoffs, n x the passage's own
    tokens per word - head for word cutoffs."""
    if kind == "tokens":
        return MNT_CUTOFF_FACTOR * (n - head_tokens)
    return n * (tokens / words) - head_tokens if words > 0 else 0.0


def length_budget(requested: float, cutoffs: dict[str, list[int]], tokens: int, words: int, rnd: int,
                  head_tokens: int = 0, cap: int | None = None) -> tuple[int, dict[str, list[int]], int]:
    """(max_new_tokens, cutoffs kept, number of cutoffs dropped).

    max_new_tokens = ``requested`` (the level's factor x the tokens it asks for) raised to what every served
    cutoff needs (``cutoff_need``), then ceiled to a multiple of ``rnd``. The served cutoffs can exceed the
    passage (the acceptance slack serves hi 1500 tokens from 1363), so a budget scaled only on the passage
    can fall short of them. Cutoffs are dropped only when the budget would exceed ``cap`` (context limit
    minus prompt tokens): then every cutoff needing more than the capped budget goes, and the budget is
    min(cap rounded down, what the rest needs)."""
    def budget(cuts: dict[str, list[int]]) -> int:
        need = max([float(requested)] + [cutoff_need(k, n, tokens, words, head_tokens)
                                         for k in KINDS for n in cuts.get(k, [])])
        mnt = ceil_to(need, rnd)
        while mnt < need:            # guard against float rounding in ceil_to
            mnt += rnd
        return mnt

    kept = {k: list(cutoffs.get(k, [])) for k in KINDS}
    mnt = budget(kept)
    if cap is None or mnt <= cap:
        return mnt, kept, 0
    cap_r = max(rnd, cap // rnd * rnd)
    kept = {k: [n for n in v if cutoff_need(k, n, tokens, words, head_tokens) <= cap_r] for k, v in kept.items()}
    dropped = sum(len(cutoffs.get(k, [])) - len(kept[k]) for k in KINDS)
    return min(cap_r, budget(kept)), kept, dropped


_PLACEHOLDER = re.compile(r"\{(text|head|points|title|prompt|n)\}")


def render(template: str, **values: Any) -> str:
    """Fill {text} {head} {points} {title} {prompt} {n} in ONE pass, so braces inside the inserted human text
    are never touched. A placeholder without a value is an error."""
    def sub(m: re.Match[str]) -> str:
        k = m.group(1)
        if k not in values:
            raise KeyError(f"template needs {{{k}}}")
        return str(values[k])
    return _PLACEHOLDER.sub(sub, template)


def script_share(text: str, language: str) -> float:
    """Share of letters (L*, M*) in the language's script; the same count as trim.script_share (0.0 for a
    language without a script entry)."""
    script = {"hi": "DEVANAGARI", "en": "LATIN"}.get(language)
    letters = hit = 0
    for ch in text:
        if unicodedata.category(ch)[0] in "LM":
            letters += 1
            if script and unicodedata.name(ch, "").startswith(script):
                hit += 1
    return hit / letters if letters else 0.0


# --------------------------------------------------------------------------- config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """edit_levels.yaml, with cutoffs / tokens_per_word / min_frac / slack_words filled in from promptset.yaml
    and the kit loaded. ``lengths["slack"]`` is the resolved slack ({"words": S, "tokens": {lang: T}}, see
    trim.resolve_slack). Returns a plain dict."""
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    ps = yaml.safe_load((base / cfg.get("promptset_config", "promptset.yaml")).read_text(encoding="utf-8"))
    lengths = dict(cfg.get("lengths") or {})
    lengths.setdefault("cutoffs", ps["cutoffs"])
    lengths.setdefault("tokens_per_word", ps["tokens_per_word"])
    lengths.setdefault("min_frac", ps.get("min_frac", 0.95))
    lengths.setdefault("slack_words", ps.get("slack_words", DEFAULT_SLACK_WORDS))
    lengths.setdefault("slack_tokens", None)          # None: round(slack_words * tokens_per_word[lang])
    lengths.setdefault("max_new_tokens_round", ps.get("max_new_tokens_round", 64))
    lengths.setdefault("round_n_to", 10)
    lengths.setdefault("max_context", 32768)        # Qwen3-8B native context (validate_promptset --max-context)
    lengths["cutoffs"] = {k: sorted(int(v) for v in lengths["cutoffs"].get(k, [])) for k in KINDS}
    set_slack(lengths)
    cfg["lengths"] = lengths
    cfg["kit_data"] = json.loads((base / cfg.get("kit", "ai_levels_prompts.json")).read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    return cfg


def set_slack(lengths: dict[str, Any], slack_words: int | None = None, slack_tokens: int | None = None) -> None:
    """(Re)compute ``lengths["slack"]`` and ``lengths["acceptance_rule"]``; CLI values override the config."""
    if slack_words is not None:
        lengths["slack_words"] = int(slack_words)
    if slack_tokens is not None:
        lengths["slack_tokens"] = int(slack_tokens)
    st = lengths.get("slack_tokens")
    lengths["slack"] = resolve_slack(int(lengths["slack_words"]), lengths["tokens_per_word"],
                                     None if st is None else int(st))
    lengths["acceptance_rule"] = describe_rule(float(lengths["min_frac"]), lengths["slack"],
                                               int(lengths["slack_words"]) if st is None else None)


def kit_level(cfg: dict[str, Any], level_id: str) -> dict[str, Any]:
    for lv in cfg["kit_data"]["levels"]:
        if lv["id"] == level_id:
            return lv
    raise KeyError(f"level {level_id!r} not in the prompt kit")


def templates_for(cfg: dict[str, Any], level: str, lang: str) -> list[tuple[int, str]]:
    """(variant index, template) pairs for a level. Index = position in the kit's (or our) list."""
    lc = cfg["levels"][level]
    if level == "OX_extract_outline":
        return list(enumerate(cfg["ours"]["OX_extract_outline"][lang]))
    kit_id = lc.get("kit_level", level)
    variants = kit_level(cfg, kit_id)[lang]
    idx = lc.get("variants") or list(range(len(variants)))
    return [(i, variants[i]) for i in idx]


def source_cfg(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    out = dict(cfg["sources"].get("default") or {})
    out.update(cfg["sources"].get(source) or {})
    return out


# --------------------------------------------------------------------------- token counting


class TokenCounter:
    """Qwen3 token counts (``tokenizer(text, add_special_tokens=False)``, as trim.py and the verifier do) or,
    without a tokenizer, an estimate ``round(words * tokens_per_word[lang])`` (``estimated`` is then True)."""

    def __init__(self, fn: Callable[[str], int] | None, tokens_per_word: dict[str, float], name: str = "estimate"):
        self.fn = fn
        self.tpw = tokens_per_word
        self.name = name
        self.estimated = fn is None

    def __call__(self, text: str, lang: str) -> int:
        text = text.strip()
        if self.fn is not None:
            return self.fn(text)
        return int(round(count_words(text) * float(self.tpw[lang])))

    def for_lang(self, lang: str) -> Callable[[str], int]:
        return lambda s: self(s, lang)


def load_token_counter(spec: str | None, tokens_per_word: dict[str, float]) -> TokenCounter:
    """``spec``: a local tokenizer dir/file or HF id; "auto" (default) = the staged Qwen3 tokenizer if it exists;
    "none" = estimate from tokens_per_word."""
    if spec in (None, "", "auto"):
        spec = str(DEFAULT_TOKENIZER) if DEFAULT_TOKENIZER.exists() else "none"
    if spec.lower() == "none":
        return TokenCounter(None, tokens_per_word)
    p = Path(spec)
    tj = p / "tokenizer.json" if p.is_dir() else p
    if tj.suffix == ".json" and tj.exists():
        try:
            from tokenizers import Tokenizer

            tk = Tokenizer.from_file(str(tj))
            return TokenCounter(lambda s: len(tk.encode(s, add_special_tokens=False).ids), tokens_per_word, str(tj))
        except ImportError:
            pass
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise SystemExit(f"--tokenizer {spec}: needs `pip install tokenizers` (for tokenizer.json) or transformers") from e
    tok = AutoTokenizer.from_pretrained(str(p.parent if p.suffix == ".json" else spec))
    return TokenCounter(lambda s: len(tok(s, add_special_tokens=False)["input_ids"]), tokens_per_word, spec)


# --------------------------------------------------------------------------- cutting


def _truncate_words(text: str, max_words: int) -> str:
    """``text`` up to the start of word ``max_words + 1`` (the whole text if shorter)."""
    for i, m in enumerate(re.finditer(r"\S+", text)):
        if i == max_words:
            return text[: m.start()]
    return text


def cut_passage(text: str, lang: str, cutoff_kind: str, n: int, tokenizer: TokenCounter | None,
                min_frac: float = 0.95, pattern: re.Pattern[str] = SENT_END,
                slack: dict[str, Any] | int | None = None) -> Trim:
    """Cut ``text`` at the last sentence end at or before ``n`` words/tokens; accepted iff
    units >= min(min_frac * n, n - slack).

    ``slack``: a slack spec ({"words": S, "tokens": T | {lang: T}}, as ``load_config`` puts in
    ``lengths["slack"]``), resolved for ``cutoff_kind`` and ``lang``, or a plain int; None/0 = no slack.
    This is trim.py's ``trim_text`` (same sentence ends, same rule). Long documents are first truncated
    to n + 2 words (a passage of <= n tokens has <= n words, as every word is at least one token; the
    estimate allows for tokens_per_word < 1); the truncation can only add a candidate end that is longer than
    n, so the result is identical to trimming the full text, only faster.
    """
    if cutoff_kind not in KINDS:
        raise ValueError(f"cutoff kind must be one of {KINDS}")
    counter = tokenizer.for_lang(lang) if tokenizer is not None else None
    if cutoff_kind == "tokens":
        if counter is None:
            raise ValueError("token cutoffs need a TokenCounter")
        tpw = float(tokenizer.tpw.get(lang, 1.0)) if tokenizer.estimated else 1.0
        bound = int(math.floor((n + 0.5) / min(1.0, tpw))) + 1
    else:
        bound = n + 1
    sl = slack if isinstance(slack, int) else slack_for(slack, cutoff_kind, lang)
    return trim_text(_truncate_words(text, bound + 1), cutoff_kind, n, counter, min_frac, pattern, sl)


def largest_cut(text: str, lang: str, kind: str, cutoffs: Sequence[int], counter: TokenCounter,
                min_frac: float, pattern: re.Pattern[str], slack: dict[str, Any] | int | None = None
                ) -> tuple[int, Trim] | None:
    """The largest cutoff of ``kind`` the text supports, and its passage."""
    for n in sorted(cutoffs, reverse=True):
        t = cut_passage(text, lang, kind, n, counter, min_frac, pattern, slack)
        if t.accepted:
            return n, t
    return None


def served_cutoffs(passage: str, lang: str, cutoffs: dict[str, list[int]], counter: TokenCounter,
                   min_frac: float, pattern: re.Pattern[str], slack: dict[str, Any] | int | None = None
                   ) -> dict[str, list[int]]:
    """Every configured cutoff at which the human passage itself trims to an accepted passage (all N up to the
    passage's length, under the same rule as trim.py)."""
    return {kind: [n for n in cutoffs.get(kind, [])
                   if cut_passage(passage, lang, kind, n, counter, min_frac, pattern, slack).accepted]
            for kind in KINDS}


def continuation_head(passage: str, k: int, pattern: re.Pattern[str]) -> tuple[str, int] | None:
    """The first k % of the passage's words, cut at the sentence end nearest to k % (ties: the shorter head).
    None if the passage has no inner sentence end."""
    total = count_words(passage)
    target = total * k / 100.0
    word_starts = [m.start() for m in re.finditer(r"\S+", passage)]
    best: tuple[float, int, int] | None = None
    for e in sentence_ends(passage, pattern):
        w = bisect.bisect_left(word_starts, e)
        if 0 < w < total:
            cand = (abs(w - target), w, e)
            if best is None or cand < best:
                best = cand
    if best is None:
        return None
    head = passage[: best[2]].strip()
    return head, count_words(head)


# --------------------------------------------------------------------------- headers and titles


_SEPARATOR_LINE = re.compile(r"^\s*[_\-*=~•.·]{3,}\s*$")
_HEADING_END_OK = re.compile(f"[{re.escape(_TERMINALS)},;][{_CLOSERS}]*\\s*$")


def strip_header(text: str, max_lines: int = 8, max_words: int = 10, fixed_lines: int = 0) -> tuple[str, int, str]:
    """Drop leading heading/byline lines (titles, author names, picture credits, "____" separators).

    0. Drop the first ``fixed_lines`` non-empty lines unconditionally: a known header, such as the title and
       author lines of hindisamay and medium files (``sources.<name>.header_lines``; titles can be long).
    1. If a separator line ("____", "----", "***") is among the next ``max_lines`` non-empty lines, drop
       everything up to and including the last such separator.
    2. Then drop leading lines of <= ``max_words`` words that do not end in sentence or clause punctuation
       (. ! ? । ॥ … , ;), at most ``max_lines`` of them in total (after the fixed lines).
    Returns (rest of the text, number of dropped non-empty lines, the dropped text).
    """
    lines = text.split("\n")
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    fixed = nonempty[:max(0, int(fixed_lines))]
    start = fixed[-1] + 1 if fixed else 0
    dropped = 0
    nonempty = [i for i in nonempty if i >= start]
    window = nonempty[:max_lines]
    seps = [i for i in window if _SEPARATOR_LINE.match(lines[i])]
    if seps:
        start = seps[-1] + 1
        dropped = sum(1 for i in window if i <= seps[-1])
    for i in [j for j in nonempty if j >= start]:
        ln = lines[i].strip()
        if dropped >= max_lines or count_words(ln) > max_words or _HEADING_END_OK.search(ln) \
                or ln[:1] in "\"“‘'":
            break
        start = i + 1
        dropped += 1
    dropped += len(fixed)
    if start == 0:
        return text, 0, ""
    rest = "\n".join(lines[start:])
    if not rest.strip():  # nothing but header: keep the text as it is
        return text, 0, ""
    return rest, dropped, "\n".join(lines[:start]).strip()


def header_lines(doc: "SourceDoc", sc: dict[str, Any]) -> int:
    """Known header lines (title, author) to drop before the heuristic: ``sources.<name>.header_lines``, else the
    corpus's ``meta["header_lines"]``, else 0."""
    v = sc.get("header_lines")
    return int(doc.meta.get("header_lines", 0) if v is None else v)


def usable_title(title: str | None, lang: str, check: dict[str, Any]) -> tuple[bool, str]:
    """(ok, reason). Rejects empty titles, slugs or Latin text for Hindi, section labels and prose lines."""
    t = (title or "").strip()
    if not t:
        return False, "empty"
    nw = count_words(t)
    if nw < int(check.get("min_words", 1)) or nw > int(check.get("max_words", 14)):
        return False, f"{nw} words"
    if any(c in t for c in check.get("reject_chars", "")):
        return False, "label/slug characters"
    if re.search(r"[:：]\s*$", t):
        return False, "ends with a colon (label)"
    if re.search(r"[.।॥]\s*$", t) and not re.search(r"(?:\.\.\.|…)\s*$", t):   # a trailing ellipsis is a title style
        return False, "ends like a sentence"
    if sum(1 for ch in t if unicodedata.category(ch)[0] == "L") < 2:
        return False, "too few letters"
    share = script_share(t, lang)
    need = float((check.get("min_script_share") or {}).get(lang, 0.9))
    if share < need:
        return False, f"script share {share:.2f}"
    return True, "ok"


# --------------------------------------------------------------------------- documents


@dataclass
class SourceDoc:
    pair_id: str          # human doc id, unique across sources ("<source>/<id>")
    source: str
    source_id: str
    language: str
    title: str
    text: str
    genre: str
    license: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return count_words(self.text)


def load_corpus_docs(name: str, cfg: dict[str, Any], normalize: str | None = None) -> list[SourceDoc]:
    """Documents of a registered corpus (corpora package, repo root on sys.path)."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import corpora

    sc = source_cfg(cfg, name)
    lang = sc.get("language")
    if not lang:
        raise SystemExit(f"config sources.{name}.language is not set")
    try:
        corpus = corpora.get(name, normalize=normalize)
    except FileNotFoundError as e:
        hint = ""
        if name in ("hindisamay", "medium"):
            hint = ("; HindiSamay and Medium are not in git: git fetch origin add-hindi-corpora && "
                    "git checkout FETCH_HEAD -- data/HindiSamay data/Medium && "
                    "git restore --staged data/HindiSamay data/Medium (see promptset/EDIT_LEVELS.md)")
        raise SystemExit(f"{e}{hint}") from e
    out = []
    for d in corpus:
        meta = dict(d.meta)
        meta.setdefault("punctuation_removed", bool(sc.get("punctuation_removed", False)))
        out.append(SourceDoc(pair_id=d.id, source=name, source_id=d.id, language=meta.get("language", lang),
                             title=d.title, text=d.text, genre=str(meta.get("genre") or "unknown"),
                             license=str(meta.get("license") or sc.get("license") or "see source"), meta=meta))
    return out


JSONL_FIELDS = ("id", "language", "text")


def load_jsonl_docs(path: str | Path, cfg: dict[str, Any], source: str | None = None) -> list[SourceDoc]:
    """Generic documents: one JSON object per line with id, language ("en"|"hi"), text, and optionally title,
    genre, license, source, meta (dict), punctuation_removed (bool)."""
    out = []
    src_default = source or Path(path).stem
    for i, r in enumerate(iter_jsonl(path), 1):
        missing = [f for f in JSONL_FIELDS if not r.get(f)]
        if missing:
            raise SystemExit(f"{path}:{i}: missing field(s) {missing}")
        src = str(r.get("source") or src_default)
        sc = source_cfg(cfg, src)
        meta = dict(r.get("meta") or {})
        meta["punctuation_removed"] = bool(r.get("punctuation_removed", meta.get(
            "punctuation_removed", sc.get("punctuation_removed", False))))
        rid = str(r["id"])
        out.append(SourceDoc(pair_id=f"{src}/{rid}", source=src, source_id=rid, language=str(r["language"]),
                             title=str(r.get("title") or ""), text=str(r["text"]),
                             genre=str(r.get("genre") or "unknown"),
                             license=str(r.get("license") or sc.get("license") or "see source"), meta=meta))
    return out


# --------------------------------------------------------------------------- passages


@dataclass
class Passage:
    id: str
    doc: SourceDoc
    experiment: str            # first experiment it serves
    experiments: list[str]     # all experiments it serves (two when the words and tokens cuts coincide)
    cutoff_kind: str
    cutoff: int
    cut_by: dict[str, int]     # kind -> cutoff, for every experiment it serves
    text: str
    words: int
    tokens: int
    cutoffs: dict[str, list[int]]
    char_start: int            # offset of the passage in doc.text
    header_lines_stripped: int


_MARKUP = re.compile(r"<!|\[endif\]|\[if [a-z]|</?(?:p|div|span|br|a|img)\b[^>]*>|&(?:nbsp|amp|lt|gt);")


SCRIPT_FAIL = "script share"
SCRIPT_LABEL = {"hi": "Devanagari", "en": "Latin"}


def passage_script_min(cfg: dict[str, Any]) -> dict[str, float]:
    """Minimum share of the language's script (Devanagari for hi, Latin for en) among a passage's letters:
    trim.py's DEFAULT_SCRIPT_MIN (hi 0.8, en 0.9), the threshold its script check applies to the same human
    passages, unless ``sampling.script_min`` overrides it per language."""
    out = dict(DEFAULT_SCRIPT_MIN)
    out.update({str(k): float(v) for k, v in ((cfg.get("sampling") or {}).get("script_min") or {}).items()})
    return out


def passage_id(doc: SourceDoc, exp: str) -> str:
    return f"HP-{doc.source}-{short_hash(doc.pair_id)}-{EXP_SHORT.get(exp, exp)}-{doc.language}"


def make_passages(doc: SourceDoc, cfg: dict[str, Any], counter: TokenCounter
                  ) -> tuple[list[Passage], dict[str, str]]:
    """Cut the doc once per experiment at the largest cutoff it supports. Returns (passages, failures)."""
    L = cfg["lengths"]
    sc = source_cfg(cfg, doc.source)
    pr = bool(doc.meta.get("punctuation_removed", False))
    pattern = sentence_pattern(pr)
    body, n_hdr = doc.text, 0
    if sc.get("strip_header"):
        body, n_hdr, _ = strip_header(doc.text, fixed_lines=header_lines(doc, sc))
    lead = len(body) - len(body.lstrip())
    base = len(doc.text) - len(body) + lead
    script_min = passage_script_min(cfg)
    out: list[Passage] = []
    fails: dict[str, str] = {}
    for exp, kind in (cfg.get("experiments") or {"same_words": "words", "same_tokens": "tokens"}).items():
        res = largest_cut(body, doc.language, kind, L["cutoffs"][kind], counter, L["min_frac"], pattern,
                          L.get("slack"))
        if res is None:
            fails[exp] = f"no accepted {kind} cut (doc has {doc.n_words} words)"
            continue
        n, t = res
        same = next((p for p in out if p.text == t.passage), None)
        if same is not None:   # identical passage: one set of rows serves both experiments
            same.experiments.append(exp)
            same.cut_by[kind] = n
            continue
        text = t.passage
        if _MARKUP.search(text):
            fails[exp] = "HTML/markup left in the passage"
            continue
        need = script_min.get(doc.language)
        if need is not None:
            share = script_share(text, doc.language)
            if share < need:
                fails[exp] = f"{SCRIPT_FAIL} {share:.2f} < {need:g}"
                continue
        assert doc.text[base: base + len(text)] == text, "passage must be a prefix of the doc body"
        out.append(Passage(
            id=passage_id(doc, exp), doc=doc, experiment=exp, experiments=[exp], cutoff_kind=kind, cutoff=n,
            cut_by={kind: n}, text=text, words=count_words(text), tokens=counter(text, doc.language),
            cutoffs=served_cutoffs(text, doc.language, L["cutoffs"], counter, L["min_frac"], pattern,
                                   L.get("slack")),
            char_start=base, header_lines_stripped=n_hdr))
    return out, fails


# --------------------------------------------------------------------------- sampling


def sample_docs(docs: Sequence[SourceDoc], n: int | None, seed: int, accept: Callable[[SourceDoc], bool],
                stratify_by: str | None = "genre") -> list[SourceDoc]:
    """Stable sample of up to ``n`` accepted docs: docs in stable-hash order (seeded), round-robin over the
    strata (genres) so that each stratum gets an equal share while it lasts. ``accept`` is evaluated lazily."""
    order = sorted(docs, key=lambda d: (stable_hash(seed, "sample", d.pair_id), d.pair_id))
    strata: dict[str, list[SourceDoc]] = defaultdict(list)
    for d in order:
        strata[getattr(d, stratify_by, "all") if stratify_by else "all"].append(d)
    queues = {k: iter(v) for k, v in sorted(strata.items())}
    chosen: list[SourceDoc] = []
    while queues and (n is None or len(chosen) < n):
        for key in list(queues):
            if n is not None and len(chosen) >= n:
                break
            for d in queues[key]:
                if accept(d):
                    chosen.append(d)
                    break
            else:
                del queues[key]
    return chosen


def apportion(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``n`` into integer parts proportional to ``weights`` (largest remainder; ties by key), summing to n."""
    w = {k: float(v) for k, v in weights.items() if float(v) > 0}
    if n <= 0 or not w:
        return {k: 0 for k in weights}
    tot = sum(w.values())
    raw = {k: n * v / tot for k, v in w.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    for k in sorted(w, key=lambda k: (-(raw[k] - out[k]), k))[: n - sum(out.values())]:
        out[k] += 1
    return {k: out.get(k, 0) for k in weights}


def genre_mix_plan(cfg: dict[str, Any], lang: str, sources: Sequence[str], n: int) -> dict[str, dict[str, int]]:
    """Quota per genre and source for one language: ``n`` x ``sampling.genre_mix`` per genre, each split over
    the sources by ``sampling.genre_sources[lang][genre]`` (shares), restricted to the loaded ``sources``. A
    genre without configured shares for any loaded source is split equally over them."""
    samp = cfg.get("sampling") or {}
    mix = samp.get("genre_mix") or {}
    if not mix:
        raise SystemExit("genre_mix sampling needs sampling.genre_mix in the config, e.g. {story: 0.5, essay: 0.5}")
    per_genre = apportion(n, mix)
    plan: dict[str, dict[str, int]] = {}
    shares_cfg = ((samp.get("genre_sources") or {}).get(lang) or {})
    for g, q in per_genre.items():
        shares = {s: float(v) for s, v in (shares_cfg.get(g) or {}).items() if s in sources and float(v) > 0}
        if not shares:
            shares = {s: 1.0 for s in sources}
        plan[g] = apportion(q, dict(sorted(shares.items())))
    return plan


def sample_genre_mix(docs: Sequence[SourceDoc], plan: dict[str, dict[str, int]], seed: int,
                     accept: Callable[[SourceDoc], bool]) -> tuple[list[SourceDoc], dict[tuple[str, str], dict[str, int]]]:
    """Genre-matched sample of one language. ``plan``: genre -> source -> quota (``genre_mix_plan``).

    Each (genre, source) cell takes its quota of accepted docs in the same seeded stable-hash order as
    ``sample_docs`` (so a cell's picks are a prefix of its order and grow stably with n). If a source runs out
    of usable docs, the other sources of that genre fill the gap (largest share first). Docs whose genre is
    not in the plan are never used. Returns (docs, per-cell counts), the docs interleaved round-robin over the
    cells (genres in plan order, then sources), so that the first K (pilot) docs keep the mix."""
    order = sorted(docs, key=lambda d: (stable_hash(seed, "sample", d.pair_id), d.pair_id))
    cells: dict[tuple[str, str], list[SourceDoc]] = defaultdict(list)
    for d in order:
        cells[(d.genre, d.source)].append(d)
    picked: dict[tuple[str, str], list[SourceDoc]] = {}
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for (g, s), v in cells.items():
        counts[(g, s)] = {"available": len(v), "quota": int((plan.get(g) or {}).get(s, 0)), "sampled": 0}
    for g, quotas in plan.items():
        its = {s: iter(cells.get((g, s), [])) for s in quotas}
        short = 0
        for s, q in quotas.items():
            got = picked.setdefault((g, s), [])
            while len(got) < q:
                d = next(its[s], None)
                if d is None:
                    break
                if accept(d):
                    got.append(d)
            short += q - len(got)
            counts.setdefault((g, s), {"available": 0, "quota": q, "sampled": 0})
        for s in sorted(quotas, key=lambda s: (-quotas[s], s)):     # top up from the other sources of the genre
            got = picked[(g, s)]
            while short > 0:
                d = next(its[s], None)
                if d is None:
                    break
                if accept(d):
                    got.append(d)
                    short -= 1
    for k, v in picked.items():
        counts[k]["sampled"] = len(v)
    queues = [picked[(g, s)] for g, quotas in plan.items() for s in quotas if picked.get((g, s))]
    chosen: list[SourceDoc] = []
    for i in range(max((len(q) for q in queues), default=0)):
        chosen.extend(q[i] for q in queues if i < len(q))
    return chosen, counts


# --------------------------------------------------------------------------- rows


def row_id(level: str, doc: SourceDoc, exp: str, k: int | None = None) -> str:
    lv = f"{level}_k{k}" if k is not None else level
    return f"{lv}-{doc.source}-{short_hash(doc.pair_id)}-{EXP_SHORT.get(exp, exp)}-{doc.language}"


def pick_variant(cfg: dict[str, Any], level: str, pair_id: str, lang: str) -> tuple[int, str]:
    """Kit phrasing for (doc, level): stable hash of the doc id, so it is the same in every experiment and
    for every k, and rotates across docs."""
    opts = templates_for(cfg, level, lang)
    return opts[stable_hash(cfg.get("seed", 0), "template", level, pair_id) % len(opts)]


def sentence_end_rule(doc: SourceDoc) -> str:
    """"punct" = trim.py's SENT_END; "punct+linebreak" = also every line break (punctuation-stripped text).
    trim.py must use the same rule for these rows' human passages and for C rows' head + continuation."""
    return "punct+linebreak" if doc.meta.get("punctuation_removed") else "punct"


def base_row(p: Passage, level: str, split: str, counter: TokenCounter, k: int | None = None) -> dict[str, Any]:
    d = p.doc
    return {
        "id": row_id(level, d, p.experiment, k),
        "pair_id": d.pair_id,
        "language": d.language,
        "level": level,
        "genre": d.genre,
        "source": d.source,
        "source_id": d.source_id,
        "license": d.license,
        "split": split,
        "experiment": p.experiment,
        "experiments": list(p.experiments),
        "passage_id": p.id,
        "input_cutoff_kind": p.cutoff_kind,
        "input_cutoff": p.cutoff,
        "input_cut_by": dict(p.cut_by),
        "input_words": p.words,
        "input_tokens": p.tokens,
        "token_counts_estimated": counter.estimated,
        "sentence_end_rule": sentence_end_rule(d),
        "human_passage": p.text,
        "k": k,
    }


def finish_row(row: dict[str, Any], cfg: dict[str, Any], level: str, variant: int, user: str,
               target_words: int | None, max_new_tokens: int, cutoffs: dict[str, list[int]],
               watermark: bool, meta: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    lang = row["language"]
    row.update(extra or {})
    row.update({
        "target_words": target_words,
        "max_new_tokens": int(max_new_tokens),
        "cutoffs": {k: list(v) for k, v in cutoffs.items()},
        "template_id": f"{level}/{variant}",
        "messages": [{"role": "system", "content": cfg["kit_data"]["system"][lang]},
                     {"role": "user", "content": user}],
        "chat_template_kwargs": {"enable_thinking": False},
        "generation": {"seed": seed_of(row["id"])},
        "watermark": watermark,
        "meta": meta,
    })
    return row


def doc_meta(p: Passage, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    m = {"punctuation_removed": bool(p.doc.meta.get("punctuation_removed", False)),
         "header_lines_stripped": p.header_lines_stripped, "passage_char_start": p.char_start,
         "doc_words": p.doc.n_words}
    for k in ("genre_fine", "genre_rule"):        # the corpus's own genre label / how a genre was assigned
        if p.doc.meta.get(k):
            m[k] = p.doc.meta[k]
    m.update(extra or {})
    return m


def level_allowed(cfg: dict[str, Any], level: str, doc: SourceDoc) -> bool:
    if doc.meta.get("punctuation_removed") and cfg["levels"][level].get("punctuation_removed", "flag") == "exclude":
        return False
    return True


def resolve_title(doc: SourceDoc, cfg: dict[str, Any], overrides: dict[str, str | None]) -> tuple[str | None, str]:
    """(title or None, where it came from / why it was rejected)."""
    if doc.pair_id in overrides:
        t = overrides[doc.pair_id]
        return (t.strip(), "override") if t and t.strip() else (None, "override: skip")
    sc = source_cfg(cfg, doc.source)
    check = {**(cfg.get("title_check") or {}), **(sc.get("title_check") or {})}   # per-source overrides
    max_chars = sc.get("title_max_chars") or {}
    reasons = []
    for src in sc.get("title_from") or ["title"]:
        if src == "header_lines":   # the heading lines strip_header removes, first usable one wins
            hdr = strip_header(doc.text, fixed_lines=header_lines(doc, sc))[2]
            cands = [ln.strip() for ln in hdr.split("\n") if ln.strip() and not _SEPARATOR_LINE.match(ln)]
        else:
            cands = [doc.title if src == "title" else doc.meta.get(src)]
        why = "none"
        for cand in cands:
            ok, why = usable_title(cand, doc.language, check)
            if ok and src in max_chars and len(cand.strip()) > int(max_chars[src]):
                ok, why = False, f"{len(cand.strip())} chars (slug probably cut short)"
            if ok:
                return cand.strip(), src
        reasons.append(f"{src}: {why}")
    return None, "; ".join(reasons) or "no candidates"


def build_rows_for_passage(p: Passage, cfg: dict[str, Any], levels: Sequence[str], split: str,
                           counter: TokenCounter, title: str | None, title_src: str,
                           skips: Counter) -> list[dict[str, Any]]:
    L = cfg["lengths"]
    rnd, rn = int(L["max_new_tokens_round"]), int(L["round_n_to"])
    max_context = L.get("max_context")
    d, lang = p.doc, p.doc.language
    pattern = sentence_pattern(bool(d.meta.get("punctuation_removed")))
    system = cfg["kit_data"]["system"][lang]
    rows: list[dict[str, Any]] = []

    def scored_row(level: str, k: int | None, v: int, user: str, target_words: int | None, requested: float,
                   head_tokens: int, meta: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """A watermarked row whose budget covers every cutoff it serves (see ``length_budget``)."""
        cap = None
        if max_context:
            cap = int(max_context) - counter(system + "\n" + user, lang) - CHAT_TEMPLATE_TOKENS
        mnt, cuts, dropped = length_budget(requested, p.cutoffs, p.tokens, p.words, rnd, head_tokens, cap)
        if dropped:
            skips[(level, d.source, "cutoffs dropped: over max_context")] += dropped
        return finish_row(base_row(p, level, split, counter, k), cfg, level, v, user, target_words, mnt, cuts,
                          True, meta, extra)

    for level in levels:
        if level == "O_outline":
            continue  # stage 2: build-outline
        if not level_allowed(cfg, level, d):
            skips[(level, d.source, "punctuation_removed: excluded")] += 1
            continue
        lc = cfg["levels"][level]
        kind = lc.get("kind")
        if kind == "edit":
            v, tpl = pick_variant(cfg, level, d.pair_id, lang)
            rows.append(scored_row(level, None, v, render(tpl, text=p.text), p.words,
                                   lc["max_new_tokens_factor"] * p.tokens, 0, doc_meta(p)))
        elif kind == "continue":
            v, tpl = pick_variant(cfg, level, d.pair_id, lang)
            for k in lc.get("ks", [75, 50, 25]):
                h = continuation_head(p.text, int(k), pattern)
                if h is None:
                    skips[(level, d.source, "no inner sentence end")] += 1
                    continue
                head, hw = h
                ht = counter(head, lang)
                n = round_to(p.words - hw, rn)
                rows.append(scored_row(
                    level, int(k), v, render(tpl, head=head, n=n), n,
                    lc["max_new_tokens_factor"] * max(1, p.tokens - ht), ht, doc_meta(p),
                    {"head": head, "head_words": hw, "head_tokens": ht, "head_share": round(hw / p.words, 4),
                     "continuation_words_human": p.words - hw, "continuation_tokens_human": p.tokens - ht}))
        elif kind == "title":
            if title is None:
                skips[(level, d.source, "no usable title")] += 1
                continue
            v, tpl = pick_variant(cfg, level, d.pair_id, lang)
            n = round_to(p.words, rn)
            rows.append(scored_row(level, None, v, render(tpl, title=title, n=n), n,
                                   lc["max_new_tokens_factor"] * p.tokens, 0,
                                   doc_meta(p, {"title_source": title_src}), {"title": title}))
        elif kind == "extract_outline":
            v, tpl = pick_variant(cfg, level, d.pair_id, lang)
            mnt = lc["max_new_tokens"][lang] if isinstance(lc["max_new_tokens"], dict) else lc["max_new_tokens"]
            rows.append(finish_row(base_row(p, level, split, counter), cfg, level, v, render(tpl, text=p.text),
                                   None, int(mnt), {"words": [], "tokens": []}, False, doc_meta(p),
                                   {"passage_cutoffs": p.cutoffs}))
        else:
            raise ValueError(f"level {level}: unknown kind {kind!r}")
    return rows


def passage_record(p: Passage, counter: TokenCounter, split: str) -> dict[str, Any]:
    d = p.doc
    return {
        "id": p.id, "pair_id": d.pair_id, "language": d.language, "level": "human", "genre": d.genre,
        "source": d.source, "source_id": d.source_id, "license": d.license, "split": split,
        "experiment": p.experiment, "experiments": list(p.experiments), "input_cutoff_kind": p.cutoff_kind,
        "input_cutoff": p.cutoff, "input_cut_by": dict(p.cut_by), "n_words": p.words, "n_tokens": p.tokens,
        "token_counts_estimated": counter.estimated,
        "sentence_end_rule": sentence_end_rule(d),
        "cutoffs": p.cutoffs, "title": d.title, "meta": doc_meta(p), "text": p.text,
    }


# --------------------------------------------------------------------------- build


def read_title_overrides(path: str | None) -> dict[str, str | None]:
    """JSONL rows {"pair_id": ..., "title": ... | null} or TSV lines "pair_id<TAB>title" (empty = skip doc)."""
    if not path:
        return {}
    out: dict[str, str | None] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.lstrip().startswith("{"):
            r = json.loads(line)
            out[str(r["pair_id"])] = r.get("title")
        else:
            pid, _, t = line.partition("\t")
            if pid == "pair_id":
                continue
            out[pid] = t.split("\t")[0] if t else None
    return out


def resolve_levels(cfg: dict[str, Any], levels: Sequence[str] | None) -> list[str]:
    names = list(levels) if levels else list(cfg["levels"])
    bad = [n for n in names if n not in cfg["levels"]]
    if bad:
        raise SystemExit(f"unknown level(s) {bad}; known: {list(cfg['levels'])}")
    if "O_outline" in names and "OX_extract_outline" not in names:
        names.append("OX_extract_outline")   # O needs stage 1
    return [lv for lv in LEVEL_ORDER if lv in names] + [lv for lv in names if lv not in LEVEL_ORDER]


def build(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    if getattr(args, "slack_words", None) is not None or getattr(args, "slack_tokens", None) is not None:
        set_slack(cfg["lengths"], getattr(args, "slack_words", None), getattr(args, "slack_tokens", None))
    if args.experiments:
        cfg["experiments"] = {e: cfg["experiments"][e] for e in args.experiments}
    L = cfg["lengths"]
    counter = load_token_counter(args.tokenizer, L["tokens_per_word"])
    log(f"token counts: {'ESTIMATED from tokens_per_word ' + str(L['tokens_per_word']) if counter.estimated else counter.name}"
        f"; sentence ends from {TRIM_SOURCE}; {L['acceptance_rule']}")
    levels = resolve_levels(cfg, args.levels)
    samp = cfg.get("sampling") or {}
    named = bool(args.corpus or args.jsonl)
    mode = args.sampling or ("per_corpus" if named else str(samp.get("mode", "genre_mix")))
    if mode == "genre_mix":
        n_docs = args.n_docs if args.n_docs is not None else samp.get("n_docs_per_language")
        if not n_docs or int(n_docs) <= 0:
            raise SystemExit("genre_mix sampling needs a number of docs per language > 0 "
                             "(sampling.n_docs_per_language or --n-docs)")
        n_docs = int(n_docs)
    else:
        n_docs = args.n_docs if args.n_docs is not None else samp.get("n_docs")
        n_docs = None if n_docs is not None and int(n_docs) <= 0 else n_docs
    min_words = args.min_words if args.min_words is not None else int(samp.get("min_words", 0))
    pilot_docs = args.pilot_docs if args.pilot_docs is not None else int(samp.get("pilot_docs", 0))
    require_all = bool(samp.get("require_all_experiments", True))
    overrides = read_title_overrides(args.titles)
    seed = int(cfg.get("seed", 0))

    corpus_names = list(args.corpus or [])
    if not named:       # the study build: every source named under sampling.genre_sources
        corpus_names = sorted({src for by_genre in (samp.get("genre_sources") or {}).values()
                               for shares in (by_genre or {}).values() for src in (shares or {})})
    groups: dict[tuple[str, str], list[SourceDoc]] = defaultdict(list)
    for name in corpus_names:
        for d in load_corpus_docs(name, cfg, args.normalize):
            groups[(d.source, d.language)].append(d)
    for path in args.jsonl or []:
        for d in load_jsonl_docs(path, cfg, args.jsonl_source):
            groups[(d.source, d.language)].append(d)
    if not groups:
        raise SystemExit("no input: give --corpus NAME and/or --jsonl FILE, or set sampling.genre_sources")

    cache: dict[str, tuple[list[Passage], dict[str, str]]] = {}
    reasons: dict[str, Counter] = defaultdict(Counter)

    def accept(d: SourceDoc) -> bool:
        why = reasons[d.source]
        if d.n_words < min_words:
            why[f"< {min_words} words"] += 1
            return False
        ps, fails = make_passages(d, cfg, counter)
        cache[d.pair_id] = (ps, fails)
        if not ps or (require_all and fails):
            script = sorted(e for e, w in fails.items() if w.startswith(SCRIPT_FAIL))
            if script:   # a script failure is reported on its own (the doc is replaced like any other)
                need = passage_script_min(cfg).get(d.language)
                why[f"passage {SCRIPT_FAIL} < {need:g}: " + "; ".join(script)] += 1
            else:
                why["no passage: " + "; ".join(sorted(fails))[:80]] += 1
            return False
        return True

    selections: list[tuple[str, list[SourceDoc]]] = []     # (group label, docs in sample order)
    doc_counts: dict[str, dict[str, int]] = {}
    plans: dict[str, dict[str, dict[str, int]]] = {}
    if mode == "genre_mix":
        by_lang: dict[str, list[SourceDoc]] = defaultdict(list)
        for (_, lang), docs in sorted(groups.items()):
            by_lang[lang].extend(docs)
        for lang, docs in sorted(by_lang.items()):
            plan = genre_mix_plan(cfg, lang, sorted({d.source for d in docs}), n_docs)
            plans[lang] = plan
            chosen, counts = sample_genre_mix(docs, plan, seed, accept)
            for (g, src), c in sorted(counts.items()):
                doc_counts[f"{lang}/{g}/{src}"] = c
            for g, quotas in plan.items():
                got = sum(counts.get((g, src), {}).get("sampled", 0) for src in quotas)
                if got < sum(quotas.values()):
                    log(f"warning: {lang}/{g}: only {got} of {sum(quotas.values())} docs usable (genre mix not met)")
            selections.append((lang, chosen))
    else:
        for (source, lang), docs in sorted(groups.items()):
            chosen = sample_docs(docs, n_docs, seed, accept, samp.get("stratify_by", "genre"))
            doc_counts[f"{source}/{lang}"] = {"available": len(docs), "sampled": len(chosen)}
            if n_docs is not None and len(chosen) < n_docs:
                log(f"warning: {source}/{lang}: only {len(chosen)} of {n_docs} docs usable")
            selections.append((f"{source}/{lang}", chosen))

    rows: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    titles: list[list[Any]] = []
    skips: Counter = Counter()
    for source, why in sorted(reasons.items()):
        for reason, v in why.items():
            skips[("(doc)", source, reason)] += v
    for _, chosen in selections:
        for i, d in enumerate(chosen):
            split = "pilot" if i < pilot_docs else args.split
            title, tsrc = resolve_title(d, cfg, overrides) if "G_title" in levels else (None, "")
            if "G_title" in levels:
                titles.append([d.pair_id, d.source, d.language, d.genre, title or "", tsrc, d.title,
                               str(d.meta.get("first_line", ""))[:80]])
            for p in cache[d.pair_id][0]:
                passages.append(passage_record(p, counter, split))
                rows.extend(build_rows_for_passage(p, cfg, levels, split, counter, title, tsrc, skips))

    ids = [r["id"] for r in rows]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        raise SystemExit(f"duplicate row ids (bug): {dup[:5]}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "edit_prompts.jsonl", rows)
    write_jsonl(out / "human_passages.jsonl", passages)
    if titles:
        with open(out / "titles_review.tsv", "w", encoding="utf-8") as f:
            f.write("pair_id\tsource\tlanguage\tgenre\ttitle_used\ttitle_from\tdoc_title\tfirst_line\n")
            for t in titles:
                f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in t) + "\n")
    meta = {
        "command": "build", "argv": sys.argv[1:], "config": cfg["_path"], "trim_source": TRIM_SOURCE,
        "tokenizer": counter.name, "token_counts_estimated": counter.estimated,
        "acceptance_rule": L["acceptance_rule"], "lengths": L,
        "experiments": cfg["experiments"], "levels": levels, "sampling_mode": mode,
        "n_docs": n_docs, "n_docs_unit": "per language" if mode == "genre_mix" else "per source and language",
        "genre_mix": samp.get("genre_mix") if mode == "genre_mix" else None,
        "genre_plan": plans or None, "min_words": min_words,
        "script_min": passage_script_min(cfg),
        "docs_skipped_script": sum(v for (a, _, c), v in skips.items() if a == "(doc)" and SCRIPT_FAIL in c),
        "docs": doc_counts,
        "n_rows": len(rows), "n_passages": len(passages),
        "skips": [{"level": a, "source": b, "reason": c, "n": v} for (a, b, c), v in sorted(skips.items())],
    }
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    md = summary_markdown(rows, passages, meta)
    (out / "edit_prompts_summary.md").write_text(md, encoding="utf-8")
    print(md)
    log(f"wrote {out / 'edit_prompts.jsonl'} ({len(rows)} rows), {out / 'human_passages.jsonl'} ({len(passages)})")
    return meta


def level_label(r: dict[str, Any]) -> str:
    return f"{r['level']}_k{r['k']}" if r.get("k") is not None else r["level"]


def summary_markdown(rows: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]], meta: dict[str, Any]) -> str:
    def exp_label(r: dict[str, Any]) -> str:
        return "+".join(r.get("experiments") or [r["experiment"]])

    lines = ["# Edit-level prompt set", ""]
    lines.append(f"{len(rows)} prompt rows, {len(passages)} human passages. Token counts: "
                 + ("**estimated** from tokens_per_word (no tokenizer)" if meta["token_counts_estimated"]
                    else f"Qwen3 tokenizer ({meta['tokenizer']})") + ".")
    if meta.get("acceptance_rule"):
        lines += ["", f"Length rule: {meta['acceptance_rule']}."]
    if meta.get("script_min"):
        lines += ["", "Passage script check (trim.py's threshold): a document is skipped and replaced by the next "
                  "one in the sample if a passage has less than "
                  + ", ".join(f"{v:g} {SCRIPT_LABEL.get(k, k)} ({k})" for k, v in sorted(meta["script_min"].items()))
                  + f" among its letters; {meta.get('docs_skipped_script', 0)} skipped."]
    if meta.get("lengths", {}).get("max_new_tokens_round"):
        Lm = meta["lengths"]
        lines += ["", f"max_new_tokens = ceil{Lm['max_new_tokens_round']}(max(level factor x requested tokens, "
                  f"{MNT_CUTOFF_FACTOR} x (largest served token cutoff - head tokens), largest served word cutoff x "
                  f"passage tokens/word - head tokens)); prompt + max_new_tokens <= {Lm.get('max_context')}."]
    if meta.get("sampling_mode") == "genre_mix":
        lines += ["", "## Documents (genre-matched sample)", "",
                  f"{meta['n_docs']} documents per language, genre mix {meta.get('genre_mix')}; quota per genre and "
                  "source from sampling.genre_sources (a source short of usable documents is topped up by the "
                  "other sources of its genre). Genres outside the mix are not sampled (quota 0).", "",
                  md_table(["lang/genre/source", "available", "quota", "sampled"],
                           [[k, v["available"], v.get("quota", "-"), v["sampled"]] for k, v in meta["docs"].items()])]
    else:
        lines += ["", "## Documents", "", md_table(["source/lang", "available", "sampled"],
                                                   [[k, v["available"], v["sampled"]] for k, v in meta["docs"].items()])]
    dg = Counter((lang, g, src) for lang, g, src, _ in {(p["language"], p["genre"], p["source"], p["pair_id"])
                                                         for p in passages})
    lines += ["", "## Sampled documents per language x genre x source", "",
              md_table(["lang", "genre", "source", "docs"], [list(k) + [v] for k, v in sorted(dg.items())])]
    lg = Counter()
    for r in rows:
        for e in r.get("experiments") or [r["experiment"]]:
            lg[(level_label(r), r["language"], r["genre"], e)] += 1
    lines += ["", "## Rows serving each experiment, per level x language x genre", "",
              "A row whose words and tokens cuts coincide serves both experiments and is counted under each "
              "(filter on `experiments`).", "",
              md_table(["level", "lang", "genre", "experiment", "rows"], [list(k) + [v] for k, v in sorted(lg.items())])]
    c = Counter((level_label(r), exp_label(r), r["source"], r["language"]) for r in rows)
    lines += ["", "## Rows per level x experiment x source x language", "",
              md_table(["level", "experiment", "source", "lang", "rows"], [list(k) + [v] for k, v in sorted(c.items())])]
    pc = Counter((exp_label(p), p["source"], p["language"], p["input_cutoff_kind"], p["input_cutoff"]) for p in passages)
    lines += ["", "## Passages per experiment and input cutoff", "",
              md_table(["experiment", "source", "lang", "cut by", "cutoff", "passages"],
                       [list(k) + [v] for k, v in sorted(pc.items(), key=lambda kv: tuple(map(str, kv[0])))])]
    served = Counter()
    for p in passages:
        for kind in KINDS:
            for n in p["cutoffs"][kind]:
                served[(exp_label(p), p["source"], p["language"], kind, n)] += 1
    lines += ["", "## Passages serving each cutoff (human anchor exists at that cutoff)", "",
              md_table(["experiment", "source", "lang", "kind", "N", "passages"],
                       [list(k) + [v] for k, v in sorted(served.items(), key=lambda kv: tuple(map(str, kv[0])))])]
    cov_tot = Counter()
    cov = Counter()
    for p in passages:
        for e in p.get("experiments") or [p["experiment"]]:
            cov_tot[(p["language"], e)] += 1
            for kind in KINDS:
                for n in p["cutoffs"][kind]:
                    cov[(p["language"], e, kind, n)] += 1
    cfg_cuts = (meta.get("lengths") or {}).get("cutoffs") or {}
    cov_rows = [[lang, e, kind, n, cov[(lang, e, kind, n)], tot, f"{cov[(lang, e, kind, n)] / tot:.0%}"]
                for (lang, e), tot in sorted(cov_tot.items()) for kind in KINDS for n in cfg_cuts.get(kind, [])]
    lines += ["", "## Served-cutoff coverage per language", "",
              "Passages (serving each experiment) whose human anchor exists at cutoff N, out of all passages of that "
              "language and experiment.", "",
              md_table(["lang", "experiment", "kind", "N", "passages", "of", "share"], cov_rows)]
    tok = Counter()
    for r in rows:
        tok[(r["level"], r["language"])] += r["max_new_tokens"]
    lines += ["", "## Upper bound on generated tokens (sum of max_new_tokens)", "",
              md_table(["level", "lang", "tokens"], [list(k) + [f"{v:,}"] for k, v in sorted(tok.items())])]
    per_lang: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in rows:
        per_lang[(r["language"], str(r.get("split")))].append(r["max_new_tokens"])
    lines += ["", "## max_new_tokens per language and split", "",
              md_table(["lang", "split", "rows", "mean", "max", "sum"],
                       [[l, sp, len(v), f"{sum(v) / len(v):,.0f}", f"{max(v):,}", f"{sum(v):,}"]
                        for (l, sp), v in sorted(per_lang.items())])]
    if meta["skips"]:
        lines += ["", "## Skipped", "", md_table(["level", "source", "reason", "n"],
                                                 [[s["level"], s["source"], s["reason"], s["n"]] for s in meta["skips"]])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- build-outline


_BULLET = re.compile(r"^\s*(?:[-*•●▪◦‣–—+]|\(?(?:\d{1,2}|[०-९]{1,2})[.)।:]|\(?[a-zA-Z][.)])\s+(.*\S)\s*$")


def clean_outline(text: str, max_bullets: int = 8, truncated: bool = False) -> list[str]:
    """Bullet lines only (-, *, •, 1., 1), १.), markers and **bold** removed, capped at ``max_bullets``. If the
    generation hit max_new_tokens (``truncated``), the last bullet may be cut off and is dropped."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    bullets = []
    for line in text.splitlines():
        m = _BULLET.match(line)
        if not m:
            continue
        b = m.group(1).replace("**", "").replace("__", "").strip()
        b = re.sub(r"^[-*•]\s+", "", b).strip()
        if count_words(b) == 0 or re.fullmatch(r"[\W_]+", b):
            continue
        bullets.append(b)
    if truncated and bullets:
        bullets = bullets[:-1]
    return bullets[:max_bullets]


OUTPUT_FIELDS = ("text", "token_ids", "n_new_tokens", "finish_reason", "had_think", "attempt", "watermarked",
                 "seed_effective", "batch_seed", "batch_size", "prompt_tokens", "max_new_tokens_effective",
                 "min_new_tokens_effective", "model_id", "config_sha", "seconds", "gen_time_utc")


def build_outline_rows(gen_rows: Iterable[dict[str, Any]], cfg: dict[str, Any]
                       ) -> tuple[list[dict[str, Any]], Counter]:
    """O_outline rows from generate.py outputs of OX rows (input fields + "text"). Per OX id, the lowest
    attempt with >= min_bullets bullets is used."""
    lc = cfg["levels"]["O_outline"]
    L = cfg["lengths"]
    rnd, rn = int(L["max_new_tokens_round"]), int(L["round_n_to"])
    lo, hi = int(lc.get("min_bullets", 3)), int(lc.get("max_bullets", 8))
    est = TokenCounter(None, L["tokens_per_word"])
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats: Counter = Counter()
    for r in gen_rows:
        if r.get("level") != "OX_extract_outline":
            stats["not an OX row"] += 1
            continue
        by_id[str(r["id"])].append(r)
    rows = []
    for ox_id, attempts in sorted(by_id.items()):
        attempts.sort(key=lambda r: int(r.get("attempt") or 0))
        chosen = None
        for r in attempts:
            pts = clean_outline(r.get("text", ""), hi, truncated=r.get("finish_reason") == "length")
            if len(pts) >= lo:
                chosen = (r, pts)
                break
        if chosen is None:
            stats[f"dropped: < {lo} bullets"] += 1
            continue
        r, pts = chosen
        lang = r["language"]
        level = "O_outline"
        doc_like = SourceDoc(pair_id=r["pair_id"], source=r["source"], source_id=r["source_id"], language=lang,
                             title="", text="", genre=r.get("genre", "unknown"), license=r.get("license", ""))
        rid = row_id(level, doc_like, r["experiment"])
        row = {k: v for k, v in r.items() if k not in OUTPUT_FIELDS and k not in (
            "messages", "template_id", "generation", "watermark", "cutoffs", "passage_cutoffs", "target_words",
            "max_new_tokens", "chat_template_kwargs", "meta")}
        row.update({"id": rid, "level": level, "k": None})
        v, tpl = pick_variant(cfg, level, r["pair_id"], lang)
        n = round_to(int(r["input_words"]), rn)
        points = "\n".join(f"- {b}" for b in pts)
        meta = dict(r.get("meta") or {})
        meta.update({"outline_model": r.get("model_id"), "outline_finish_reason": r.get("finish_reason"),
                     "outline_watermarked": r.get("watermarked")})
        user = render(tpl, points=points, n=n)
        cap = None
        if L.get("max_context"):   # prompt tokens estimated (the O prompt is short: the outline only)
            cap = int(L["max_context"]) - est(cfg["kit_data"]["system"][lang] + "\n" + user, lang) \
                - CHAT_TEMPLATE_TOKENS
        mnt, cuts, dropped = length_budget(lc["max_new_tokens_factor"] * int(r["input_tokens"]),
                                           r.get("passage_cutoffs") or {"words": [], "tokens": []},
                                           int(r["input_tokens"]), int(r["input_words"]), rnd, 0, cap)
        if dropped:
            stats["cutoffs dropped: over max_context"] += dropped
        finish_row(row, cfg, level, v, user, n, mnt, cuts, True, meta,
                   {"points": points, "n_bullets": len(pts), "outline_source_id": ox_id,
                    "outline_attempt": int(r.get("attempt") or 0)})
        if r.get("watermarked"):
            stats["warning: OX generated WITH watermark"] += 1
        rows.append(row)
        stats["O rows"] += 1
    return rows, stats


def build_outline(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config)
    gen = read_jsonl_files(args.gen)
    rows, stats = build_outline_rows(gen, cfg)
    out = Path(args.out)
    write_jsonl(out, rows)
    c = Counter((r["experiment"], r["source"], r["language"]) for r in rows)
    md = ["# O_outline rows", "", f"{len(rows)} rows from {len(gen)} generation rows ({args.gen}).", "",
          md_table(["experiment", "source", "lang", "rows"], [list(k) + [v] for k, v in sorted(c.items())]),
          "", md_table(["event", "n"], [[k, v] for k, v in sorted(stats.items())])]
    text = "\n".join(md) + "\n"
    out.with_name(out.name[: -len(out.suffix)] + "_summary.md" if out.suffix else out.name + "_summary.md"
                  ).write_text(text, encoding="utf-8")
    print(text)
    return {"n_rows": len(rows), "stats": dict(stats)}


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="cut human passages and write edit_prompts.jsonl + human_passages.jsonl")
    b.add_argument("--config", default=str(DEFAULT_CONFIG))
    b.add_argument("--corpus", action="append",
                   help="registered corpus name (repeatable), e.g. samalochan; default (none given): the sources "
                        "under sampling.genre_sources (hindisamay, samalochan, medium)")
    b.add_argument("--jsonl", action="append", help="generic JSONL docs: id, language, title, text, genre, license")
    b.add_argument("--jsonl-source", help="source name for --jsonl rows without a 'source' field (default: file stem)")
    b.add_argument("--sampling", choices=("genre_mix", "per_corpus"),
                   help="genre_mix: one genre-matched sample per language (sampling.n_docs_per_language x "
                        "sampling.genre_mix, sources from sampling.genre_sources); per_corpus: n docs per source. "
                        "Default: genre_mix without --corpus/--jsonl (the study build), per_corpus with them")
    b.add_argument("--n-docs", type=int,
                   help="genre_mix: docs per language (default sampling.n_docs_per_language); per_corpus: docs per "
                        "source and language (default sampling.n_docs; 0 = all)")
    b.add_argument("--levels", nargs="+", help="subset of levels (O_outline implies OX_extract_outline)")
    b.add_argument("--experiments", nargs="+", choices=sorted(EXP_SHORT), help="subset of experiments")
    b.add_argument("--min-words", type=int, help="minimum words of a source document (default: config)")
    b.add_argument("--split", default="main", help="split label of the rows (default main)")
    b.add_argument("--pilot-docs", type=int,
                   help="first K sampled docs per source (per_corpus) or per language (genre_mix) get split=pilot")
    b.add_argument("--tokenizer", default="auto",
                   help="Qwen3 tokenizer dir / tokenizer.json / HF id; 'auto' = "
                        f"{DEFAULT_TOKENIZER} if present, else estimate; 'none' = estimate")
    b.add_argument("--slack-words", type=int,
                   help="word slack of the rule n >= min(min_frac*N, N - slack), also the base of the derived token "
                        "slack (default: config lengths.slack_words / promptset.yaml slack_words, 30; 0 = off)")
    b.add_argument("--slack-tokens", type=int,
                   help="token slack for every language (default: round(slack_words * tokens_per_word[lang]))")
    b.add_argument("--titles", help="title overrides for G_title: TSV pair_id<TAB>title or JSONL (empty = skip)")
    b.add_argument("--normalize", help="Unicode normalisation of corpus text (default: raw bytes)")
    b.add_argument("--out", default=str(REPO_ROOT / "promptset" / "data" / "edit"))
    o = sub.add_parser("build-outline", help="O_outline rows from generate.py outputs of the OX rows")
    o.add_argument("--config", default=str(DEFAULT_CONFIG))
    o.add_argument("--gen", nargs="+", required=True, help="generate.py output JSONL(s) holding OX rows")
    o.add_argument("--out", default=str(REPO_ROOT / "promptset" / "data" / "edit" / "O_prompts.jsonl"))
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "build":
        build(args)
    else:
        build_outline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
