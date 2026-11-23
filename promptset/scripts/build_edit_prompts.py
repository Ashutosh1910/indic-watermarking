"""Build the edit-level prompt set (L1-L4, C, OX -> O, G_title) from HUMAN documents.

Every row has the same schema as the G set (build_promptset.py), so one generate.py run handles all of
them. See promptset/EDIT_LEVELS.md for the study design and the GPU run order.

Subcommands::

    # 1) cut human passages and write the prompt rows (OX outline-extraction rows included)
    python promptset/scripts/build_edit_prompts.py build --corpus samalochan --corpus hindi-aesthetics \\
        --n-docs 150 --out promptset/data/edit [--tokenizer /path/to/qwen3_tokenizer]
    #    generic JSONL input (one doc per line: id, language, title, text, genre, license[, source])
    python promptset/scripts/build_edit_prompts.py build --jsonl medium_en.jsonl --n-docs 150 --out ...

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

Length rule (shared with trim.py): a passage is cut at the last sentence end at or before N words or
tokens and accepted only if it keeps >= min_frac * N. Sentence ends come from trim.py (``SENT_END``);
for punctuation-stripped corpora (hindi-aesthetics) every line break is also a sentence end.
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
    from trim import _CLOSERS, _TERMINALS, SENT_END, Trim, count_words, sentence_ends, trim_text  # noqa: E402
    TRIM_SOURCE = "trim.py"
except ImportError:  # pragma: no cover - fallback; keep in sync with trim.py
    TRIM_SOURCE = "local fallback"
    _CLOSERS = "\"'”’)\\]»"
    _TERMINALS = ".!?।॥…"
    SENT_END = re.compile(f"[{re.escape(_TERMINALS)}]+[{_CLOSERS}]*(?=\\s|$)")

    @dataclass
    class Trim:  # type: ignore[no-redef]
        passage: str
        n_words: int
        n_tokens: int | None
        accepted: bool
        reject_reason: str | None
        full_units: int | None

    def count_words(text: str) -> int:  # type: ignore[no-redef]
        return len(text.split())

    def sentence_ends(text: str, pattern: re.Pattern[str] = SENT_END) -> list[int]:  # type: ignore[no-redef]
        return [m.end() for m in pattern.finditer(text)]

    def trim_text(text, kind, cutoff, count_tokens, min_frac=0.95, pattern=SENT_END):  # type: ignore[no-redef]
        if not text.strip():
            return Trim("", 0, 0 if count_tokens else None, False, "empty", 0)
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
            return Trim("", 0, 0 if count_tokens else None, False, "no_sentence_end", full)
        p = text[:best].strip()
        nw, nt = count_words(p), (count_tokens(p) if count_tokens else None)
        units = nw if kind == "words" else nt
        return Trim(p, nw, nt, units >= min_frac * cutoff, None if units >= min_frac * cutoff else "too_short", full)

CONFIG_DIR = REPO_ROOT / "promptset" / "config"
DEFAULT_CONFIG = CONFIG_DIR / "edit_levels.yaml"
DEFAULT_TOKENIZER = Path("/mnt/project-files/promptset_stage/qwen3_tokenizer")
LEVEL_ORDER = ("L1_proofread", "L2_polish", "L3_paraphrase", "L4_rewrite", "C_continue", "G_title",
               "OX_extract_outline", "O_outline")
EXP_SHORT = {"same_words": "sw", "same_tokens": "st"}
KINDS = ("words", "tokens")

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
    """edit_levels.yaml, with cutoffs / tokens_per_word / min_frac filled in from promptset.yaml and the kit
    loaded. Returns a plain dict."""
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent
    ps = yaml.safe_load((base / cfg.get("promptset_config", "promptset.yaml")).read_text(encoding="utf-8"))
    lengths = dict(cfg.get("lengths") or {})
    lengths.setdefault("cutoffs", ps["cutoffs"])
    lengths.setdefault("tokens_per_word", ps["tokens_per_word"])
    lengths.setdefault("min_frac", ps.get("min_frac", 0.95))
    lengths.setdefault("max_new_tokens_round", ps.get("max_new_tokens_round", 64))
    lengths.setdefault("round_n_to", 10)
    lengths["cutoffs"] = {k: sorted(int(v) for v in lengths["cutoffs"].get(k, [])) for k in KINDS}
    cfg["lengths"] = lengths
    cfg["kit_data"] = json.loads((base / cfg.get("kit", "ai_levels_prompts.json")).read_text(encoding="utf-8"))
    cfg["_path"] = str(path)
    return cfg


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
                min_frac: float = 0.95, pattern: re.Pattern[str] = SENT_END) -> Trim:
    """Cut ``text`` at the last sentence end at or before ``n`` words/tokens; accepted iff >= min_frac * n.

    This is trim.py's ``trim_text`` (same sentence ends, same 95 % rule). Long documents are first truncated
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
    return trim_text(_truncate_words(text, bound + 1), cutoff_kind, n, counter, min_frac, pattern)


def largest_cut(text: str, lang: str, kind: str, cutoffs: Sequence[int], counter: TokenCounter,
                min_frac: float, pattern: re.Pattern[str]) -> tuple[int, Trim] | None:
    """The largest cutoff of ``kind`` the text supports, and its passage."""
    for n in sorted(cutoffs, reverse=True):
        t = cut_passage(text, lang, kind, n, counter, min_frac, pattern)
        if t.accepted:
            return n, t
    return None


def served_cutoffs(passage: str, lang: str, cutoffs: dict[str, list[int]], counter: TokenCounter,
                   min_frac: float, pattern: re.Pattern[str]) -> dict[str, list[int]]:
    """Every configured cutoff at which the human passage itself trims to an accepted passage (all N up to the
    passage's length, under the same rule as trim.py)."""
    return {kind: [n for n in cutoffs.get(kind, [])
                   if cut_passage(passage, lang, kind, n, counter, min_frac, pattern).accepted]
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


def strip_header(text: str, max_lines: int = 8, max_words: int = 10) -> tuple[str, int, str]:
    """Drop leading heading/byline lines (titles, author names, picture credits, "____" separators).

    1. If a separator line ("____", "----", "***") is among the first ``max_lines`` non-empty lines, drop
       everything up to and including the last such separator.
    2. Then drop leading lines of <= ``max_words`` words that do not end in sentence or clause punctuation
       (. ! ? । ॥ … , ;), at most ``max_lines`` of them in total.
    Returns (rest of the text, number of dropped non-empty lines, the dropped text).
    """
    lines = text.split("\n")
    nonempty = [i for i, ln in enumerate(lines) if ln.strip()]
    start = 0
    dropped = 0
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
    if start == 0:
        return text, 0, ""
    rest = "\n".join(lines[start:])
    if not rest.strip():  # nothing but header: keep the text as it is
        return text, 0, ""
    return rest, dropped, "\n".join(lines[:start]).strip()


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
    if re.search(r"[.।॥]\s*$", t):
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
    out = []
    for d in corpora.get(name, normalize=normalize):
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
        body, n_hdr, _ = strip_header(doc.text)
    lead = len(body) - len(body.lstrip())
    base = len(doc.text) - len(body) + lead
    out: list[Passage] = []
    fails: dict[str, str] = {}
    for exp, kind in (cfg.get("experiments") or {"same_words": "words", "same_tokens": "tokens"}).items():
        res = largest_cut(body, doc.language, kind, L["cutoffs"][kind], counter, L["min_frac"], pattern)
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
        assert doc.text[base: base + len(text)] == text, "passage must be a prefix of the doc body"
        out.append(Passage(
            id=passage_id(doc, exp), doc=doc, experiment=exp, experiments=[exp], cutoff_kind=kind, cutoff=n,
            cut_by={kind: n}, text=text, words=count_words(text), tokens=counter(text, doc.language),
            cutoffs=served_cutoffs(text, doc.language, L["cutoffs"], counter, L["min_frac"], pattern),
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
    check = cfg.get("title_check") or {}
    sc = source_cfg(cfg, doc.source)
    max_chars = sc.get("title_max_chars") or {}
    reasons = []
    for src in sc.get("title_from") or ["title"]:
        if src == "header_lines":   # the heading lines strip_header removes, first usable one wins
            hdr = strip_header(doc.text)[2]
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
    d, lang = p.doc, p.doc.language
    pattern = sentence_pattern(bool(d.meta.get("punctuation_removed")))
    rows: list[dict[str, Any]] = []
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
            rows.append(finish_row(base_row(p, level, split, counter), cfg, level, v, render(tpl, text=p.text),
                                   p.words, ceil_to(lc["max_new_tokens_factor"] * p.tokens, rnd), p.cutoffs,
                                   True, doc_meta(p)))
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
                rows.append(finish_row(
                    base_row(p, level, split, counter, int(k)), cfg, level, v, render(tpl, head=head, n=n), n,
                    ceil_to(lc["max_new_tokens_factor"] * max(1, p.tokens - ht), rnd), p.cutoffs, True,
                    doc_meta(p),
                    {"head": head, "head_words": hw, "head_tokens": ht, "head_share": round(hw / p.words, 4),
                     "continuation_words_human": p.words - hw, "continuation_tokens_human": p.tokens - ht}))
        elif kind == "title":
            if title is None:
                skips[(level, d.source, "no usable title")] += 1
                continue
            v, tpl = pick_variant(cfg, level, d.pair_id, lang)
            n = round_to(p.words, rn)
            rows.append(finish_row(base_row(p, level, split, counter), cfg, level, v,
                                   render(tpl, title=title, n=n), n,
                                   ceil_to(lc["max_new_tokens_factor"] * p.tokens, rnd), p.cutoffs, True,
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
    if args.experiments:
        cfg["experiments"] = {e: cfg["experiments"][e] for e in args.experiments}
    L = cfg["lengths"]
    counter = load_token_counter(args.tokenizer, L["tokens_per_word"])
    log(f"token counts: {'ESTIMATED from tokens_per_word ' + str(L['tokens_per_word']) if counter.estimated else counter.name}"
        f"; sentence ends from {TRIM_SOURCE}")
    levels = resolve_levels(cfg, args.levels)
    samp = cfg.get("sampling") or {}
    n_docs = args.n_docs if args.n_docs is not None else samp.get("n_docs")
    n_docs = None if n_docs is not None and int(n_docs) <= 0 else n_docs
    min_words = args.min_words if args.min_words is not None else int(samp.get("min_words", 0))
    pilot_docs = args.pilot_docs if args.pilot_docs is not None else int(samp.get("pilot_docs", 0))
    require_all = bool(samp.get("require_all_experiments", True))
    overrides = read_title_overrides(args.titles)
    seed = int(cfg.get("seed", 0))

    groups: dict[tuple[str, str], list[SourceDoc]] = defaultdict(list)
    for name in args.corpus or []:
        for d in load_corpus_docs(name, cfg, args.normalize):
            groups[(d.source, d.language)].append(d)
    for path in args.jsonl or []:
        for d in load_jsonl_docs(path, cfg, args.jsonl_source):
            groups[(d.source, d.language)].append(d)
    if not groups:
        raise SystemExit("no input: give --corpus NAME and/or --jsonl FILE")

    rows: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    titles: list[list[Any]] = []
    skips: Counter = Counter()
    doc_counts: dict[tuple[str, str], dict[str, int]] = {}
    for (source, lang), docs in sorted(groups.items()):
        cache: dict[str, tuple[list[Passage], dict[str, str]]] = {}
        reasons: Counter = Counter()

        def accept(d: SourceDoc) -> bool:
            if d.n_words < min_words:
                reasons[f"< {min_words} words"] += 1
                return False
            ps, fails = make_passages(d, cfg, counter)
            cache[d.pair_id] = (ps, fails)
            if not ps or (require_all and fails):
                reasons["no passage: " + "; ".join(sorted(fails))[:80]] += 1
                return False
            return True

        chosen = sample_docs(docs, n_docs, seed, accept, samp.get("stratify_by", "genre"))
        doc_counts[(source, lang)] = {"available": len(docs), "sampled": len(chosen)}
        for (reason, v) in reasons.items():
            skips[("(doc)", source, reason)] += v
        if n_docs is not None and len(chosen) < n_docs:
            log(f"warning: {source}/{lang}: only {len(chosen)} of {n_docs} docs usable")
        for i, d in enumerate(chosen):
            split = "pilot" if i < pilot_docs else args.split
            title, tsrc = resolve_title(d, cfg, overrides) if "G_title" in levels else (None, "")
            if "G_title" in levels:
                titles.append([d.pair_id, d.source, d.language, title or "", tsrc, d.title,
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
            f.write("pair_id\tsource\tlanguage\ttitle_used\ttitle_from\tdoc_title\tfirst_line\n")
            for t in titles:
                f.write("\t".join(str(x).replace("\t", " ").replace("\n", " ") for x in t) + "\n")
    meta = {
        "command": "build", "argv": sys.argv[1:], "config": cfg["_path"], "trim_source": TRIM_SOURCE,
        "tokenizer": counter.name, "token_counts_estimated": counter.estimated, "lengths": L,
        "experiments": cfg["experiments"], "levels": levels, "n_docs": n_docs, "min_words": min_words,
        "docs": {f"{s}/{l}": v for (s, l), v in doc_counts.items()},
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
    lines += ["", "## Documents", "", md_table(["source/lang", "available", "sampled"],
                                               [[k, v["available"], v["sampled"]] for k, v in meta["docs"].items()])]
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
    tok = Counter()
    for r in rows:
        tok[(r["level"], r["language"])] += r["max_new_tokens"]
    lines += ["", "## Upper bound on generated tokens (sum of max_new_tokens)", "",
              md_table(["level", "lang", "tokens"], [list(k) + [f"{v:,}"] for k, v in sorted(tok.items())])]
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
        finish_row(row, cfg, level, v, render(tpl, points=points, n=n), n,
                   ceil_to(lc["max_new_tokens_factor"] * int(r["input_tokens"]), rnd),
                   r.get("passage_cutoffs") or {"words": [], "tokens": []}, True, meta,
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
    b.add_argument("--corpus", action="append", help="registered corpus name (repeatable), e.g. samalochan")
    b.add_argument("--jsonl", action="append", help="generic JSONL docs: id, language, title, text, genre, license")
    b.add_argument("--jsonl-source", help="source name for --jsonl rows without a 'source' field (default: file stem)")
    b.add_argument("--n-docs", type=int, help="docs per source and language (default: config sampling.n_docs; 0 = all)")
    b.add_argument("--levels", nargs="+", help="subset of levels (O_outline implies OX_extract_outline)")
    b.add_argument("--experiments", nargs="+", choices=sorted(EXP_SHORT), help="subset of experiments")
    b.add_argument("--min-words", type=int, help="minimum words of a source document (default: config)")
    b.add_argument("--split", default="main", help="split label of the rows (default main)")
    b.add_argument("--pilot-docs", type=int, help="first K sampled docs per source get split=pilot")
    b.add_argument("--tokenizer", default="auto",
                   help="Qwen3 tokenizer dir / tokenizer.json / HF id; 'auto' = "
                        f"{DEFAULT_TOKENIZER} if present, else estimate; 'none' = estimate")
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
