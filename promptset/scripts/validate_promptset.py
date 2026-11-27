"""Validate model-ready prompt files (G level from build_promptset.py, edit levels from build_edit_prompts.py).

Checks, each reported with a count and up to 5 example ids (errors fail the run with --strict, warnings never do):

1. schema       required fields and types; ids unique across all files given
2. pairing      (G rows) one en + one hi row per pair_id, same genre/split/tier/target_words/source, exactly one
                untranslated row and its language == native_lang
3. balance      rows per level x split x genre x tier for en and hi (must match), Hindi-native share per genre,
                native language per split, source mix per genre, counts against config/promptset.yaml
4. script       Hindi user turns >= 80% Devanagari letters (length clause removed), English user turns contain
                no Devanagari, system prompt == the kit's for that language
5. content      on prompt_text: no length caps / short-form requests (filter_prompts.py regexes), no URLs, no
                "[ WP ]" tags or Reddit artifacts, no template placeholders, >= 5 words, exactly one length
                clause in the user turn with n == target_words, no duplicate prompt_text within a language
6. lengths      max_new_tokens >= 1.25 x largest token cutoff, word cutoffs reachable (words x tokens_per_word
                <= max_new_tokens); continuation rows subtract head_tokens, edit rows use their passage's own
                tokens/word; rows with watermark=false (OX outline extraction) may serve no cutoff; with --tokenizer: rendered prompt lengths, the think block, special tokens
                in content, prompt + max_new_tokens <= context, edit rows' input_tokens / head_tokens vs
                human_passage / head (skipped when token_counts_estimated)
7. translation  (translated rows) word-length ratio to the original outside 0.4..2.5, untranslated Latin words
                in Hindi, repeated n-gram loops

What Qwen3 inserts (checked against the Qwen3 tokenizer_config.json chat template): with
``apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)`` the
prompt is

    <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n

i.e. an *empty* think block ``<think>\\n\\n</think>\\n\\n`` right after the assistant header (4 tokens:
``<think>`` 151667, ``\\n\\n`` 271, ``</think>`` 151668, ``\\n\\n`` 271). With enable_thinking unset/True the prompt
ends at ``<|im_start|>assistant\\n`` and the model writes its own think block. So the rendered prompt must end
with that exact empty block and contain no other ``<think>``/``</think>``.

    python validate_promptset.py ../data/G_prompts.jsonl --report ../data/G_prompts_validation.md --strict
    python validate_promptset.py ../data/G_prompts.jsonl ../data/edit/*.jsonl \\
        --tokenizer /mnt/project-files/promptset_stage/qwen3_tokenizer --report ../data/validation.md

Pass G_prompts.jsonl (both languages), not also G_prompts.en/hi.jsonl: those repeat the same ids.
The JSON summary is written next to --report (same name, .json).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import filter_prompts as fp  # noqa: E402  (regexes and helpers of the rule filter)

DEFAULT_KIT = HERE.parent / "config" / "ai_levels_prompts.json"
DEFAULT_CONFIG = HERE.parent / "config" / "promptset.yaml"
LANGS = ("en", "hi")
SPLITS = ("pilot", "main", "reserve")
MAX_EXAMPLES = 5
THINK_EMPTY = "<think>\n\n</think>\n\n"
GEN_SUFFIX = "<|im_start|>assistant\n" + THINK_EMPTY
SPECIAL_STRINGS = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<tool_call>",
                   "</tool_call>", "<tool_response>", "</tool_response>")
MNT_CUTOFF_FACTOR = 1.25
RATIO_RANGE = (0.4, 2.5)
DEVA_MIN_SHARE = 0.8
MIN_TASK_WORDS = 5

# Length clause = a number followed by "words" / "शब्द" (covers "Write about {n} words.", the WritingPrompts
# premise template "a story of about {n} words", "लगभग {n} शब्दों ..."), plus "300-word" style compounds.
LEN_CLAUSE = {
    "en": re.compile(r"(?<![\w.])(\d[\d,]*)(?:\s+|-)words?\b", re.IGNORECASE),
    "hi": re.compile(r"(?<![\w.])([0-9०-९][0-9०-९,]*)\s*(?:-\s*)?शब्द"),
}
URL = re.compile(r"https?://|\bwww\.|\b[\w-]+\.(?:com|org|net|in|io|co\.uk|ly)(?:/\S*)?\b", re.IGNORECASE)
TAG = re.compile(r"[\[\(]\s*(?:[A-Z]{2,4}|डब्ल्यूपी|डब्ल्यू पी)\s*[\]\)]|<newline>|``|\br/\w+", re.UNICODE)
PLACEHOLDER = re.compile(r"\{\s*[A-Za-z_][A-Za-z0-9_]*\s*\}|\{\{|\}\}")
LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
DEVA_CHAR = re.compile(r"[ऀ-ॿ꣠-ꣿ]")
# Two sentence ends in a row ('banned.".', 'लिखिए|।'), not an ellipsis; '|' used as a danda in Hindi.
DOUBLE_END = re.compile(r"(?<!\.)[.!?।|][\"'”’)\]]*[.।](?=\s|$)")

# code -> (severity, description). Codes starting with "schema." are made on the fly.
CHECKS: dict[str, tuple[str, str]] = {
    "file.json": ("error", "line is not valid JSON or not an object"),
    "schema.id_duplicate": ("error", "id used by more than one row"),
    "pair.count": ("error", "pair_id does not have exactly one en and one hi row"),
    "pair.mismatch": ("error", "the en and hi rows of a pair differ in genre/split/length_tier/target_words/source/native_lang"),
    "pair.native": ("error", "a pair does not have exactly one translated=false row whose language == native_lang"),
    "pair.translated_flag": ("error", "translated != (language != native_lang)"),
    "balance.en_hi": ("error", "en and hi row counts differ in a level x split x genre x tier cell"),
    "balance.en_hi_edit": ("warning", "edit levels: en and hi row counts differ in a level x split x genre x experiment cell "
                                      "(a row counts in every experiment of its `experiments`)"),
    "balance.config": ("warning", "counts differ from config/promptset.yaml (per_genre, pilot_per_genre_tier)"),
    "balance.hindi_share": ("warning", "Hindi-native share of a genre is more than 0.1 off hindi_native_share"),
    "balance.split_native_mix": ("warning", "a split has no prompts of one native language"),
    "script.hi_devanagari": ("error", f"Hindi user turn has < {DEVA_MIN_SHARE:.0%} Devanagari letters"),
    "script.en_devanagari": ("error", "English user turn contains Devanagari"),
    "script.system_prompt": ("error", "system prompt differs from the kit's for this language"),
    "content.length_cap": ("error", "prompt_text still caps the length (filter_prompts *_LENGTH_CAP)"),
    "content.length_cap_soft": ("warning", "translated prompt_text matches a non-numeric length-cap pattern (often a false positive, e.g. 'छोटा सा')"),
    "content.short_form": ("error", "prompt_text asks for a short form (filter_prompts *_SHORT_FORM)"),
    "content.short_form_soft": ("warning", "translated prompt_text matches a short-form pattern (check: 'शेर' = lion, 'table', ...)"),
    "content.needs_input": ("warning", "prompt_text matches a needs-input pattern (below, the following, image, ...)"),
    "content.nsfw": ("warning", "prompt_text matches an NSFW pattern"),
    "content.url": ("error", "URL in prompt_text or the user turn"),
    "content.tag": ("error", "'[ WP ]' style tag or Reddit artifact (<newline>, ``, r/...) in prompt_text"),
    "content.placeholder": ("error", "template placeholder ({n}, {prompt}, {text}, ...) left in a message"),
    "content.too_short": ("error", f"prompt_text has fewer than {MIN_TASK_WORDS} words"),
    "content.length_clause": ("error", "user turn does not contain exactly one length clause with n == target_words"),
    "content.punct": ("warning", "user turn has a doubled sentence end ('banned.\".', 'लिखिए|।') or an ASCII '|' used as a danda"),
    "content.render": ("warning", "user turn / template_id differ from build_promptset.render_user(prompt_text)"),
    "content.duplicate": ("error", "duplicate prompt_text (normalised) within a level and language"),
    "length.mnt_vs_token_cutoff": ("error", f"max_new_tokens < {MNT_CUTOFF_FACTOR} x largest token cutoff served (minus head_tokens for continuation rows)"),
    "length.word_cutoff_unreachable": ("error", "largest word cutoff x tokens per word (the row's passage, else config) - head tokens > max_new_tokens"),
    "length.config": ("warning", "target_words / max_new_tokens / cutoffs differ from what config/promptset.yaml gives"),
    "length.no_cutoffs": ("warning", "row serves no cutoff at all"),
    "length.context": ("error", "prompt tokens + max_new_tokens exceed --max-context"),
    "tok.think_block": ("error", "rendered prompt does not end with the empty think block, or has other <think> content"),
    "tok.special_in_content": ("error", "a message contains special-token text (<|im_start|>, <think>, ...)"),
    "tok.input_tokens": ("error", "input_tokens / head_tokens do not match the tokenised human_passage / head"),
    "edit.passage_in_prompt": ("warning", "human_passage (head for continuation rows, title for title rows) not found in the user turn"),
    "edit.k": ("warning", "k is not an int in 1..99"),
    "tr.length_ratio": ("warning", f"translated / original word ratio outside {RATIO_RANGE[0]}..{RATIO_RANGE[1]}"),
    "tr.latin_leftover": ("warning", "Hindi translation keeps lowercase Latin words (untranslated leftovers; capitalised words are taken as names)"),
    "tr.repeat_loop": ("warning", "repeated n-gram loop (a token 4+ times in a row, or a 2-3-gram 3+ times)"),
    "tr.identical": ("warning", "translated prompt_text is identical to the original"),
}


# ----------------------------------------------------------------------------------------------- helpers


def is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def words(s: str) -> int:
    return len((s or "").split())


def norm_key(s: str) -> str:
    return re.sub(r"[\W_]+", " ", unicodedata.normalize("NFC", s or "").casefold()).strip()


def letters(s: str) -> list[str]:
    """Letters and combining marks (Devanagari matras and viramas are marks, not str.isalpha())."""
    return [c for c in s if unicodedata.category(c)[0] in "LM"]


def deva_share(s: str) -> float:
    ls = letters(s)
    return sum(1 for c in ls if DEVA_CHAR.match(c)) / len(ls) if ls else 0.0


def clause_numbers(text: str, lang: str) -> list[int]:
    out = []
    for m in LEN_CLAUSE[lang].finditer(text or ""):
        try:
            out.append(int(m.group(1).replace(",", "")))  # int() reads Devanagari digits too
        except ValueError:
            pass
    return out


def repeat_loop(text: str) -> str | None:
    toks = [t.strip(".,!?;:\"'()[]{}।॥-–—…").casefold() for t in (text or "").split()]
    toks = [t for t in toks if t]
    for n, need in ((1, 4), (2, 3), (3, 3)):
        for i in range(len(toks) - n * need + 1):
            gram = toks[i:i + n]
            if all(toks[i + j * n:i + (j + 1) * n] == gram for j in range(1, need)):
                return " ".join(gram)
    return None


def load_yaml(path: Path) -> dict | None:
    if not path or not Path(path).exists():
        return None
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def md_table(header: list[str], rows: list[list[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |")
    return out


# ----------------------------------------------------------------------------------------------- report


class Report:
    def __init__(self) -> None:
        self.problems: dict[str, dict[str, Any]] = {}
        self.notes: list[str] = []
        self.sections: dict[str, Any] = {}

    def add(self, code: str, ex: str, detail: str | None = None, severity: str | None = None) -> None:
        sev, desc = CHECKS.get(code, ("error", "missing or wrong type"))
        p = self.problems.setdefault(code, {"severity": severity or sev, "description": desc, "count": 0,
                                            "examples": []})
        p["count"] += 1
        if len(p["examples"]) < MAX_EXAMPLES:
            p["examples"].append(f"{ex} ({detail})" if detail else ex)

    def count(self, sev: str) -> int:
        return sum(p["count"] for p in self.problems.values() if p["severity"] == sev)

    @property
    def errors(self) -> int:
        return self.count("error")

    @property
    def warnings(self) -> int:
        return self.count("warning")


# ----------------------------------------------------------------------------------------------- loading


def load_rows(paths: list[str | Path], rep: Report) -> list[dict]:
    """Rows of every file; each gets a private '_loc' ("file:line") used when the id is unusable."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    continue
                loc = f"{Path(p).name}:{i}"
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    rep.add("file.json", loc, str(e)[:60])
                    continue
                if not isinstance(r, dict):
                    rep.add("file.json", loc, "not an object")
                    continue
                r["_loc"] = loc
                rows.append(r)
    return rows


def rid(r: dict) -> str:
    return r["id"] if is_nonempty_str(r.get("id")) else r.get("_loc", "?")


def is_g(r: dict) -> bool:
    return r.get("level") == "G"


def is_edit(r: dict) -> bool:
    return not is_g(r) and any(k in r for k in ("human_passage", "experiment", "k", "input_cutoff"))


# ----------------------------------------------------------------------------------------------- 1. schema


def check_schema(rows: list[dict], rep: Report) -> None:
    ids = Counter(r.get("id") for r in rows if is_nonempty_str(r.get("id")))
    for r in rows:
        x = rid(r)

        def need(field: str, ok: bool, detail: str | None = None) -> None:
            if not ok:
                rep.add(f"schema.{field}", x, detail)

        need("id", is_nonempty_str(r.get("id")))
        need("pair_id", is_nonempty_str(r.get("pair_id")))
        need("language", r.get("language") in LANGS, repr(r.get("language")))
        for f in ("level", "genre", "source", "license", "split"):
            need(f, is_nonempty_str(r.get(f)))
        need("max_new_tokens", is_int(r.get("max_new_tokens")) and r["max_new_tokens"] > 0, repr(r.get("max_new_tokens")))
        c = r.get("cutoffs")
        need("cutoffs", isinstance(c, dict) and all(isinstance(c.get(k), list) and all(is_int(n) and n > 0 for n in c[k])
                                                   for k in ("words", "tokens")))
        m = r.get("messages")
        ok = (isinstance(m, list) and len(m) == 2 and all(isinstance(t, dict) for t in m)
              and [t.get("role") for t in m] == ["system", "user"] and all(is_nonempty_str(t.get("content")) for t in m))
        need("messages", ok, "must be [system, user] with non-empty content")
        ctk = r.get("chat_template_kwargs")
        need("chat_template_kwargs", isinstance(ctk, dict) and ctk.get("enable_thinking") is False,
             "enable_thinking must be false")
        g = r.get("generation")
        need("generation", isinstance(g, dict) and is_int(g.get("seed")), "generation.seed int")
        need("watermark", isinstance(r.get("watermark"), bool))
        if is_g(r):
            need("split", r.get("split") in SPLITS, repr(r.get("split")))
            need("native_lang", r.get("native_lang") in LANGS)
            need("translated", isinstance(r.get("translated"), bool))
            need("length_tier", is_nonempty_str(r.get("length_tier")))
            need("target_words", is_int(r.get("target_words")) and r["target_words"] > 0)
            need("prompt_text", is_nonempty_str(r.get("prompt_text")))
            need("original_text", is_nonempty_str(r.get("original_text")))
            need("template_id", is_nonempty_str(r.get("template_id")))
        # edit-level fields: only when present
        if "human_passage" in r:
            need("human_passage", is_nonempty_str(r["human_passage"]))
        if "k" in r:
            need("k", is_int(r["k"]) or r["k"] is None)
            if is_int(r["k"]) and not 0 < r["k"] < 100:
                rep.add("edit.k", x, str(r["k"]))
        if "experiment" in r:
            need("experiment", is_nonempty_str(r["experiment"]))
        if "input_cutoff" in r:
            v = r["input_cutoff"]
            need("input_cutoff", v is None or is_int(v) or (isinstance(v, dict) and all(is_int(n) or n is None for n in v.values())))
        if "input_tokens" in r:
            need("input_tokens", is_int(r["input_tokens"]) and r["input_tokens"] >= 0)
        if "target_words" in r and not is_g(r):
            need("target_words", r["target_words"] is None or (is_int(r["target_words"]) and r["target_words"] > 0))
        if is_nonempty_str(r.get("id")) and ids[r["id"]] > 1:
            rep.add("schema.id_duplicate", x, r.get("_loc"))


def msg(r: dict, i: int) -> str:
    m = r.get("messages")
    try:
        c = m[i]["content"]
        return c if isinstance(c, str) else ""
    except (TypeError, IndexError, KeyError):
        return ""


# ----------------------------------------------------------------------------------------------- 2. pairing


PAIR_FIELDS = ("genre", "split", "length_tier", "target_words", "source", "native_lang")


def check_pairing(rows: list[dict], rep: Report) -> None:
    by = defaultdict(list)
    for r in rows:
        if is_g(r):
            by[r.get("pair_id")].append(r)
            if r.get("native_lang") in LANGS and isinstance(r.get("translated"), bool) \
                    and r["translated"] != (r.get("language") != r["native_lang"]):
                rep.add("pair.translated_flag", rid(r))
    for pid, rs in by.items():
        langs = sorted(str(r.get("language")) for r in rs)
        if langs != ["en", "hi"]:
            rep.add("pair.count", str(pid), "+".join(langs))
            continue
        en, hi = sorted(rs, key=lambda r: r["language"])
        diff = [f for f in PAIR_FIELDS if en.get(f) != hi.get(f)]
        if diff:
            rep.add("pair.mismatch", str(pid), ",".join(diff))
        native = [r for r in rs if r.get("translated") is False]
        if len(native) != 1 or native[0].get("language") != native[0].get("native_lang"):
            rep.add("pair.native", str(pid), f"{len(native)} untranslated")


# ----------------------------------------------------------------------------------------------- 3. balance


def check_balance(rows: list[dict], cfg: dict | None, rep: Report) -> dict:
    def cell(r: dict) -> tuple:
        lv = str(r.get("level")) + (f"_k{r['k']}" if is_int(r.get("k")) else "")
        tier = r.get("length_tier") or r.get("experiment") or "-"
        return lv, str(r.get("split")), str(r.get("genre")), str(tier)

    cells: Counter = Counter()
    for r in rows:
        exps = r.get("experiments")
        if is_edit(r) and isinstance(exps, list) and exps and all(is_nonempty_str(e) for e in exps):
            # an edit row whose words and tokens cuts coincide serves every experiment in `experiments`
            # (common for English): count it in each, or English looks short in the second experiment
            lv, sp, g, _ = cell(r)
            for e in dict.fromkeys(exps):
                cells[(lv, sp, g, e, r.get("language"))] += 1
        else:
            cells[(*cell(r), r.get("language"))] += 1
    g_levels = {cell(r)[0] for r in rows if is_g(r)}
    cell_rows = []
    for k in sorted({k[:4] for k in cells}):
        en, hi = cells.get((*k, "en"), 0), cells.get((*k, "hi"), 0)
        cell_rows.append([*k, en, hi, "ok" if en == hi else "MISMATCH"])
        if en != hi:
            rep.add("balance.en_hi" if k[0] in g_levels else "balance.en_hi_edit", "/".join(k), f"en={en} hi={hi}")

    # pair-level tables (G rows, one row per pair: take the native row)
    pairs = {}
    for r in rows:
        if is_g(r) and r.get("translated") is False:
            pairs.setdefault(r.get("pair_id"), r)
    target_share = (cfg or {}).get("hindi_native_share")
    share_rows, genres = [], sorted({str(p.get("genre")) for p in pairs.values()})
    for g in genres:
        ps = [p for p in pairs.values() if p.get("genre") == g]
        n_hi = sum(1 for p in ps if p.get("native_lang") == "hi")
        share = n_hi / len(ps) if ps else 0.0
        share_rows.append([g, len(ps), n_hi, len(ps) - n_hi, f"{share:.2f}", target_share if target_share is not None else "-"])
        if target_share is not None and abs(share - target_share) > 0.1:
            rep.add("balance.hindi_share", g, f"{share:.2f} vs {target_share}")
    split_native = Counter((str(p.get("split")), str(p.get("genre")), p.get("native_lang")) for p in pairs.values())
    split_rows = []
    for sp in sorted({k[0] for k in split_native}, key=lambda s: (SPLITS.index(s) if s in SPLITS else 9, s)):
        for g in genres:
            en, hi = split_native.get((sp, g, "en"), 0), split_native.get((sp, g, "hi"), 0)
            if en or hi:
                split_rows.append([sp, g, en, hi])
        en_t = sum(v for k, v in split_native.items() if k[0] == sp and k[2] == "en")
        hi_t = sum(v for k, v in split_native.items() if k[0] == sp and k[2] == "hi")
        if (en_t == 0) != (hi_t == 0):
            rep.add("balance.split_native_mix", sp, f"en-native={en_t} hi-native={hi_t}")
    src = Counter((str(p.get("genre")), str(p.get("source")), str(p.get("native_lang"))) for p in pairs.values())
    src_rows = [[g, s, nl, v] for (g, s, nl), v in sorted(src.items())]

    if cfg and pairs:
        per = cfg.get("per_genre") or {}
        pilot_n = cfg.get("pilot_per_genre_tier")
        sp_g = Counter((str(p.get("split")), str(p.get("genre"))) for p in pairs.values())
        for g in cfg.get("genres", []):
            if sp_g.get(("main", g), 0) + sp_g.get(("pilot", g), 0) != per.get("main", 0):
                rep.add("balance.config", f"main+pilot/{g}", f"{sp_g.get(('main', g), 0) + sp_g.get(('pilot', g), 0)} != {per.get('main')}")
            if sp_g.get(("reserve", g), 0) != per.get("reserve", 0):
                rep.add("balance.config", f"reserve/{g}", f"{sp_g.get(('reserve', g), 0)} != {per.get('reserve')}")
            if pilot_n is not None:
                for t in cfg.get("tiers", {}):
                    n = sum(1 for p in pairs.values() if p.get("split") == "pilot" and p.get("genre") == g and p.get("length_tier") == t)
                    if n != pilot_n:
                        rep.add("balance.config", f"pilot/{g}/{t}", f"{n} != {pilot_n}")
    return {"cells": cell_rows, "hindi_share": share_rows, "split_native": split_rows, "sources": src_rows,
            "n_pairs": len(pairs)}


# ----------------------------------------------------------------------------------------------- 4. script


def check_script(rows: list[dict], kit: dict | None, rep: Report) -> None:
    for r in rows:
        lang, x = r.get("language"), rid(r)
        user = msg(r, 1)
        if lang == "hi":
            stripped = LEN_CLAUSE["hi"].sub(" ", user)
            share = deva_share(stripped)
            if share < DEVA_MIN_SHARE:
                rep.add("script.hi_devanagari", x, f"{share:.2f}")
        elif lang == "en" and DEVA_CHAR.search(user):
            rep.add("script.en_devanagari", x, f"{len(DEVA_CHAR.findall(user))} chars")
        if kit and lang in LANGS and msg(r, 0) != kit["system"][lang]:
            rep.add("script.system_prompt", x)


# ----------------------------------------------------------------------------------------------- 5. content


def _rules(lang: str) -> dict[str, re.Pattern]:
    if lang == "hi":
        return {"length_cap": fp.HI_LENGTH_CAP, "short_form": fp.HI_SHORT_FORM, "needs_input": fp.HI_NEEDS_INPUT,
                "nsfw": fp.HI_NSFW}
    return {"length_cap": fp.EN_LENGTH_CAP, "short_form": fp.EN_SHORT_FORM, "needs_input": fp.EN_NEEDS_INPUT,
            "nsfw": fp.EN_NSFW}


def check_content(rows: list[dict], kit: dict | None, rep: Report) -> None:
    try:
        from build_promptset import render_user
    except Exception:  # noqa: BLE001  (renderer unavailable: skip the re-render check only)
        render_user = None
    seen: dict[tuple, str] = {}
    for r in rows:
        x, lang = rid(r), r.get("language")
        for i in (0, 1):
            m = PLACEHOLDER.search(msg(r, i))
            if m:
                rep.add("content.placeholder", x, m.group(0))
                break
        pt = r.get("prompt_text")
        if not isinstance(pt, str) or lang not in LANGS:
            continue
        translated = r.get("translated") is True
        for name, pat in _rules(lang).items():
            m = pat.search(pt)
            if not m:
                continue
            code = f"content.{name}"
            if name == "length_cap" and translated and not re.search(r"[0-9०-९]", m.group(0)):
                code = "content.length_cap_soft"
            elif name == "short_form" and translated:
                code = "content.short_form_soft"
            rep.add(code, x, m.group(0))
        if URL.search(pt) or (is_g(r) and URL.search(msg(r, 1))):
            rep.add("content.url", x, (URL.search(pt) or URL.search(msg(r, 1))).group(0))
        m = TAG.search(pt)
        if m:
            rep.add("content.tag", x, m.group(0))
        if words(pt) < MIN_TASK_WORDS:
            rep.add("content.too_short", x, f"{words(pt)} words")
        if is_g(r):
            nums = clause_numbers(msg(r, 1), lang)
            if len(nums) != 1 or nums[0] != r.get("target_words"):
                rep.add("content.length_clause", x, f"found {nums}, target {r.get('target_words')}")
            m = DOUBLE_END.search(msg(r, 1)) or (re.search(r"\|", msg(r, 1)) if lang == "hi" else None)
            if m:
                rep.add("content.punct", x, repr(m.group(0)))
            if render_user and kit and is_int(r.get("target_words")) and is_nonempty_str(r.get("source")):
                try:
                    exp, tid = render_user(kit, lang, r["source"], pt, r["target_words"])
                    if exp != msg(r, 1) or (r.get("template_id") and tid != r["template_id"]):
                        rep.add("content.render", x)
                except Exception as e:  # noqa: BLE001
                    rep.add("content.render", x, f"render failed: {e}")
        key = (r.get("level"), r.get("experiment"), r.get("k"), lang, norm_key(pt))
        if key in seen:
            rep.add("content.duplicate", x, f"same as {seen[key]}")
        else:
            seen[key] = x


# ----------------------------------------------------------------------------------------------- 6. lengths


def expected_g_lengths(r: dict, cfg: dict) -> dict | None:
    """What build_promptset.py derives from config for a G row (target_words, max_new_tokens, cutoffs)."""
    import math

    tier = (cfg.get("tiers") or {}).get(r.get("length_tier"))
    lang = r.get("language")
    if not tier or lang not in (cfg.get("tokens_per_word") or {}):
        return None
    req = tier["request_words"]
    req_tok = req * cfg["tokens_per_word"][lang]
    rnd = cfg["max_new_tokens_round"]
    return {"target_words": req,
            "max_new_tokens": math.ceil(cfg["max_new_tokens_factor"] * req_tok / rnd) * rnd,
            "cutoffs": {"words": [n for n in cfg["cutoffs"]["words"] if n <= cfg["serve_frac"] * req],
                        "tokens": [n for n in cfg["cutoffs"]["tokens"] if n <= cfg["serve_frac"] * req_tok]}}


def check_lengths(rows: list[dict], cfg: dict | None, rep: Report) -> None:
    tpw = (cfg or {}).get("tokens_per_word") or {"en": 1.26, "hi": 4.57}
    for r in rows:
        x, mnt, c = rid(r), r.get("max_new_tokens"), r.get("cutoffs")
        if not is_int(mnt) or not isinstance(c, dict):
            continue
        toks = [n for n in c.get("tokens") or [] if is_int(n)]
        wds = [n for n in c.get("words") or [] if is_int(n)]
        if not toks and not wds and r.get("watermark") is not False:
            rep.add("length.no_cutoffs", x)
        # Continuation rows are scored on head + continuation: only the part after the human head is generated.
        head = r["head_tokens"] if is_int(r.get("head_tokens")) else 0
        if toks and mnt < MNT_CUTOFF_FACTOR * (max(toks) - head):
            rep.add("length.mnt_vs_token_cutoff", x, f"{mnt} < {MNT_CUTOFF_FACTOR}x({max(toks)}-{head})" if head
                    else f"{mnt} < {MNT_CUTOFF_FACTOR}x{max(toks)}")
        lang = r.get("language")
        # tokens per word: the row's own passage when it has one (edit rows), else the config average
        if is_int(r.get("input_tokens")) and is_int(r.get("input_words")) and r["input_words"] > 0:
            rate = r["input_tokens"] / r["input_words"]
        else:
            rate = tpw.get(lang)
        if wds and rate and max(wds) * rate - head > mnt:
            rep.add("length.word_cutoff_unreachable", x, f"{max(wds)} words x {rate:.2f} - {head} > {mnt}")
        if cfg and is_g(r):
            exp = expected_g_lengths(r, cfg)
            if exp is None:
                rep.add("length.config", x, f"unknown tier {r.get('length_tier')!r}")
            else:
                diff = [k for k in exp if r.get(k) != exp[k]]
                if diff:
                    rep.add("length.config", x, ",".join(diff))


def load_tokenizer(path: str | None, rep: Report):
    if not path:
        rep.notes.append("Tokenizer checks skipped: no --tokenizer given.")
        return None
    if not Path(path).exists():
        rep.notes.append(f"Tokenizer checks skipped: {path} does not exist.")
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path)
    except Exception as e:  # noqa: BLE001
        rep.notes.append(f"Tokenizer checks skipped: could not load {path} ({type(e).__name__}: {e}).")
        return None


def n_tokens(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def check_tokens(rows: list[dict], tok, rep: Report, max_context: int) -> dict:
    stats: dict[tuple, list[int]] = defaultdict(list)
    sample, estimated = None, 0
    for r in rows:
        x = rid(r)
        for i in (0, 1):
            hit = next((s for s in SPECIAL_STRINGS if s in msg(r, i)), None)
            if hit:
                rep.add("tok.special_in_content", x, hit)
                break
        if not (msg(r, 0) and msg(r, 1)):
            continue
        rendered = tok.apply_chat_template([{"role": "system", "content": msg(r, 0)}, {"role": "user", "content": msg(r, 1)}],
                                           add_generation_prompt=True, tokenize=False, enable_thinking=False)
        if sample is None:
            sample = rendered[-len(GEN_SUFFIX) - 40:]
        if not rendered.endswith(GEN_SUFFIX) or rendered.count("<think>") != 1 or rendered.count("</think>") != 1:
            tail = rendered[-30:].replace("\n", "\\n")
            rep.add("tok.think_block", x, f"ends {tail!r}")
        n = n_tokens(tok, rendered)
        stats[(str(r.get("level")), str(r.get("language")))].append(n)
        if is_int(r.get("max_new_tokens")) and n + r["max_new_tokens"] > max_context:
            rep.add("length.context", x, f"{n}+{r['max_new_tokens']} > {max_context}")
        if is_edit(r) and r.get("token_counts_estimated") is True:
            estimated += 1
            continue
        for field, src in (("input_tokens", "human_passage"), ("head_tokens", "head")):
            if is_int(r.get(field)) and is_nonempty_str(r.get(src)) and is_edit(r):
                have, want = n_tokens(tok, r[src].strip()), r[field]
                if abs(have - want) > max(2, 0.01 * have):
                    rep.add("tok.input_tokens", x, f"{field}={want}, tokenised {src}={have}")
    if estimated:
        rep.notes.append(f"{estimated} edit rows have token_counts_estimated=true: input_tokens not checked.")
    table = []
    for (lv, lang), ns in sorted(stats.items()):
        table.append([lv, lang, len(ns), min(ns), int(statistics.median(ns)), max(ns)])
    return {"prompt_tokens": table, "generation_suffix_sample": sample}


def check_edit_passage(rows: list[dict], rep: Report) -> None:
    for r in rows:
        hp = r.get("human_passage")
        if not (is_edit(r) and is_nonempty_str(hp)):
            continue
        # what the template embeds: C -> {head}, G_title -> {title}, O_outline -> {points} (not checked), else {text}
        if is_nonempty_str(r.get("head")):
            probe = r["head"]
        elif is_int(r.get("k")):
            probe = hp.strip()[:60]
        elif is_nonempty_str(r.get("title")):
            probe = r["title"]
        elif str(r.get("level", "")).startswith("O_"):
            continue
        else:
            probe = hp
        if probe.strip() not in msg(r, 1):
            rep.add("edit.passage_in_prompt", rid(r))


# ----------------------------------------------------------------------------------------------- 7. translation


def check_translation(rows: list[dict], rep: Report) -> dict:
    ratios: dict[str, list[float]] = defaultdict(list)
    latin_rows, latin_words, names = 0, 0, 0
    for r in rows:
        if r.get("translated") is not True:
            continue
        pt, orig, x, lang = r.get("prompt_text"), r.get("original_text"), rid(r), r.get("language")
        if not (is_nonempty_str(pt) and is_nonempty_str(orig)):
            continue
        direction = f"{r.get('native_lang')}->{lang}"
        ratio = words(pt) / max(1, words(orig))
        ratios[direction].append(ratio)
        if not RATIO_RANGE[0] <= ratio <= RATIO_RANGE[1]:
            rep.add("tr.length_ratio", x, f"{ratio:.2f} ({words(pt)}/{words(orig)} words)")
        if norm_key(pt) == norm_key(orig):
            rep.add("tr.identical", x)
        if lang == "hi":
            lat = LATIN_WORD.findall(pt)
            low = [w for w in lat if w[0].islower()]
            names += len(lat) - len(low)
            if low:
                latin_rows += 1
                latin_words += len(low)
                rep.add("tr.latin_leftover", x, " ".join(low[:4]))
        loop = repeat_loop(pt)
        if loop:
            rep.add("tr.repeat_loop", x, loop)
    table = [[d, len(v), f"{min(v):.2f}", f"{statistics.median(v):.2f}", f"{max(v):.2f}"] for d, v in sorted(ratios.items())]
    return {"ratios": table, "latin_leftover_rows": latin_rows, "latin_leftover_words": latin_words,
            "capitalised_latin_words": names}


# ----------------------------------------------------------------------------------------------- driver


def validate(rows: list[dict], kit: dict | None = None, cfg: dict | None = None, tokenizer=None,
             rep: Report | None = None, max_context: int = 32768) -> Report:
    rep = rep or Report()
    check_schema(rows, rep)
    check_pairing(rows, rep)
    rep.sections["balance"] = check_balance(rows, cfg, rep)
    check_script(rows, kit, rep)
    check_content(rows, kit, rep)
    check_lengths(rows, cfg, rep)
    check_edit_passage(rows, rep)
    if tokenizer is not None:
        rep.sections["tokens"] = check_tokens(rows, tokenizer, rep, max_context)
    rep.sections["translation"] = check_translation(rows, rep)
    langs = Counter(r.get("language") for r in rows)
    levels = Counter(str(r.get("level")) for r in rows)
    mnt = defaultdict(list)
    for r in rows:
        if is_int(r.get("max_new_tokens")):
            mnt[(str(r.get("level")), str(r.get("language")), str(r.get("length_tier", "-")))].append(r["max_new_tokens"])
    rep.sections["overview"] = {"rows": len(rows), "languages": dict(langs), "levels": dict(levels),
                                "max_new_tokens": [[*k, len(v), min(v), max(v), sum(v)] for k, v in sorted(mnt.items())]}
    return rep


def render_markdown(rep: Report, files: list[str]) -> str:
    ov = rep.sections.get("overview", {})
    L = ["# Prompt-set validation", "",
         f"Files: {', '.join(f'`{Path(f).name}`' for f in files)}", "",
         f"Rows: {ov.get('rows', 0)} (languages {ov.get('languages')}, levels {ov.get('levels')}). "
         f"**{rep.errors} errors, {rep.warnings} warnings.**", ""]
    for n in rep.notes:
        L += [f"- {n}"]
    if rep.notes:
        L.append("")
    L += ["## Problems", ""]
    if rep.problems:
        probs = sorted(rep.problems.items(), key=lambda kv: (kv[1]["severity"] != "error", kv[0]))
        L += md_table(["severity", "check", "count", "description", "examples (up to 5)"],
                      [[p["severity"], f"`{c}`", p["count"], p["description"], "; ".join(p["examples"])] for c, p in probs])
    else:
        L.append("None.")
    b = rep.sections.get("balance", {})
    L += ["", "## Balance", "", "### Rows per level, split, genre and tier (en vs hi)", ""]
    L += md_table(["level", "split", "genre", "tier", "en", "hi", "match"], b.get("cells", []))
    if b.get("n_pairs"):
        L += ["", f"### Hindi-native share per genre (G, {b['n_pairs']} pairs)", ""]
        L += md_table(["genre", "pairs", "hi-native", "en-native", "share", "target"], b["hindi_share"])
        L += ["", "### Native language per split and genre (pairs)", ""]
        L += md_table(["split", "genre", "en-native", "hi-native"], b["split_native"])
        L += ["", "### Source mix per genre (pairs)", ""]
        L += md_table(["genre", "source", "native", "pairs"], b["sources"])
    L += ["", "## Lengths", "", "### max_new_tokens", ""]
    L += md_table(["level", "language", "tier", "rows", "min", "max", "sum"], ov.get("max_new_tokens", []))
    t = rep.sections.get("tokens")
    if t:
        L += ["", "### Rendered prompt tokens (apply_chat_template, enable_thinking=False)", ""]
        L += md_table(["level", "language", "rows", "min", "median", "max"], t["prompt_tokens"])
        if t.get("generation_suffix_sample"):
            L += ["", "End of a rendered prompt (what Qwen3 inserts with thinking off is the empty block "
                      "`<think>\\n\\n</think>\\n\\n` after the assistant header):", "",
                  "```", repr(t["generation_suffix_sample"]), "```"]
    tr = rep.sections.get("translation", {})
    if tr.get("ratios"):
        L += ["", "## Translation sanity", "", "Words in translated prompt_text / words in original_text:", ""]
        L += md_table(["direction", "rows", "min", "median", "max"], tr["ratios"])
        L += ["", f"Hindi translations with lowercase Latin words: {tr['latin_leftover_rows']} rows "
                  f"({tr['latin_leftover_words']} words); capitalised Latin words (taken as names): {tr['capitalised_latin_words']}."]
    return "\n".join(L) + "\n"


def summary_json(rep: Report, files: list[str]) -> dict:
    return {"files": [str(f) for f in files], "errors": rep.errors, "warnings": rep.warnings, "notes": rep.notes,
            "problems": rep.problems, **rep.sections}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tokenizer", help="local Qwen3 tokenizer dir (skipped if missing)")
    ap.add_argument("--report", help="markdown report path; the JSON summary goes next to it (.json)")
    ap.add_argument("--strict", action="store_true", help="exit 1 if there is any error")
    ap.add_argument("--kit", default=str(DEFAULT_KIT))
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--max-context", type=int, default=32768, help="Qwen3-8B native context (default 32768)")
    a = ap.parse_args(argv)
    rep = Report()
    kit = json.loads(Path(a.kit).read_text(encoding="utf-8")) if Path(a.kit).exists() else None
    if kit is None:
        rep.notes.append(f"Kit {a.kit} not found: system-prompt and render checks skipped.")
    cfg = load_yaml(Path(a.config))
    if cfg is None:
        rep.notes.append(f"Config {a.config} not found: config-based checks skipped.")
    rows = load_rows(a.files, rep)
    tok = load_tokenizer(a.tokenizer, rep)
    validate(rows, kit, cfg, tok, rep, a.max_context)
    md = render_markdown(rep, a.files)
    if a.report:
        out = Path(a.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(summary_json(rep, a.files), indent=2, ensure_ascii=False),
                                            encoding="utf-8")
        print(f"{len(rows)} rows: {rep.errors} errors, {rep.warnings} warnings -> {out} (+ {out.with_suffix('.json').name})")
        for c, p in sorted(rep.problems.items(), key=lambda kv: (kv[1]["severity"] != "error", kv[0])):
            print(f"  {p['severity']:7s} {c:32s} {p['count']:>5}  e.g. {', '.join(p['examples'][:3])}")
    else:
        print(md)
    return 1 if a.strict and rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
