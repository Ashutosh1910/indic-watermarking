"""Render the model-ready G-level prompt file (step 6): one chat request per prompt and language.

Inputs (under --work): selected.jsonl (select_prompts.py), translated.jsonl (translate_indictrans2.py).
Templates: config/ai_levels_prompts.json (system prompts + G templates). Lengths: config/promptset.yaml.
Output (--out dir):
    G_prompts.jsonl        every row, both languages: feed straight to generate.py
    G_prompts.en.jsonl / G_prompts.hi.jsonl
    G_prompts_summary.md   counts and examples
Each row:
    id, pair_id, language, level="G", genre, source, source_id, native_lang, translated, translation_model,
    license, split (pilot|main|reserve), length_tier, target_words, max_new_tokens, cutoffs{words,tokens},
    prompt_text (the task, in this language), original_text (as in the source, native language),
    template_id, messages [system, user], chat_template_kwargs {enable_thinking: false}, generation {seed},
    watermark: true

    python build_promptset.py --work ../work --config ../config/promptset.yaml --kit ../config/ai_levels_prompts.json --out ../data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import yaml

LENGTH_CLAUSE = {"en": "Write about {n} words.", "hi": "लगभग {n} शब्दों में लिखें।"}


def seed_of(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def tidy(text: str, lang: str) -> str:
    """Whitespace only: collapse runs of spaces in one-line prompts, ASCII '|' used as a danda -> '।'."""
    t = text.strip()
    t = re.sub(r"\s+", " ", t) if "\n" not in t else re.sub(r"[ \t]+", " ", t)
    if lang == "hi":
        t = re.sub(r"\s*\|(?=\s|$)", "।", t)
    return t


_ENDS = re.compile(r"[.!?।॥:][\"'”’)\]»]*$")


def render_user(kit: dict, lang: str, source: str, task: str, n: int) -> tuple[str, str]:
    """User turn. WritingPrompts rows are story premises -> the kit's premise template (G_generate #2).
    Every other source is already a request -> the request itself plus the kit's length clause."""
    g = next(lv for lv in kit["levels"] if lv["id"] == "G_generate")
    if source == "writingprompts":
        return g[lang][1].replace("{n}", str(n)).replace("{prompt}", task), "G_generate/premise"
    sep = "\n\n" if "\n" in task else " "
    end = task if _ENDS.search(task) else task + ("।" if lang == "hi" else ".")
    return end + sep + LENGTH_CLAUSE[lang].format(n=n), "G_generate/request+length"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(a.config))
    kit = json.load(open(a.kit, encoding="utf-8"))
    tpw = cfg["tokens_per_word"]

    sel = [json.loads(l) for l in open(work / "selected.jsonl", encoding="utf-8") if l.strip()]
    tr = {}
    for l in open(work / "translated.jsonl", encoding="utf-8"):
        if l.strip():
            r = json.loads(l)
            tr[r["id"]] = r
    review = {}
    rp = work / "translation_review.jsonl"
    if rp.exists():  # optional review: id, ok, drop, issue. Reviews only flag or drop; they never rewrite.
        for l in open(rp, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                review[r["id"]] = r

    rows, missing, dropped = [], 0, 0
    order = {"pilot": 0, "main": 1, "reserve": 2}
    sel.sort(key=lambda r: (order[r["split"]], r["genre"], r["pair_id"]))
    for s in sel:
        t = tr.get(s["pair_id"])
        if t is None:
            missing += 1
            continue
        # The translation must come from this exact source text and direction (translate_indictrans2.py
        # resumes by id, so an edited prompt could otherwise pick up an old translation).
        if t.get("text", "").strip() != s["text"].strip() or t.get("src") != s["native_lang"]:
            missing += 1
            continue
        rv = review.get(s["pair_id"], {})
        if rv.get("drop") or rv.get("ok") is False:
            dropped += 1
            continue
        translation = t["translation"]  # always the machine translation as produced, never hand-edited
        native, other = s["native_lang"], ("hi" if s["native_lang"] == "en" else "en")
        texts = {native: s["text"], other: translation}
        # Stable ids: from the pair, not a running counter, so ids and seeds survive rebuilds.
        key = hashlib.sha256(s["pair_id"].encode()).hexdigest()[:10]
        tier = cfg["tiers"][s["tier"]]
        req = tier["request_words"]
        for lang in ("en", "hi"):
            task = tidy(texts[lang], lang)
            user, tid = render_user(kit, lang, s["source"], task, req)
            req_tok = req * tpw[lang]
            mnt = math.ceil(cfg["max_new_tokens_factor"] * req_tok / cfg["max_new_tokens_round"]) * cfg["max_new_tokens_round"]
            cut_w = [n for n in cfg["cutoffs"]["words"] if n <= cfg["serve_frac"] * req]
            cut_t = [n for n in cfg["cutoffs"]["tokens"] if n <= cfg["serve_frac"] * req_tok]
            rid = f"G-{s['genre']}-{key}-{lang}"
            rows.append({
                "id": rid,
                "pair_id": s["pair_id"],
                "language": lang,
                "level": "G",
                "genre": s["genre"],
                "source": s["source"],
                "source_id": s["src_id"],
                "native_lang": native,
                "translated": lang != native,
                "translation_model": t.get("model") if lang != native else None,
                "translation_reviewed": bool(rv) if lang != native else None,
                "license": cfg["licenses"].get(s["source"], "see source"),
                "split": s["split"],
                "length_tier": s["tier"],
                "target_words": req,
                "max_new_tokens": mnt,
                "cutoffs": {"words": cut_w, "tokens": cut_t},
                "prompt_text": task,
                "original_text": s["text"],
                "template_id": tid,
                "messages": [{"role": "system", "content": kit["system"][lang]}, {"role": "user", "content": user}],
                "chat_template_kwargs": {"enable_thinking": False},
                "generation": {"seed": seed_of(rid)},
                "watermark": True,
                "source_meta": s.get("source_meta") or {},
            })

    def dump(path: Path, rs: list) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dump(out / "G_prompts.jsonl", rows)
    for lang in ("en", "hi"):
        dump(out / f"G_prompts.{lang}.jsonl", [r for r in rows if r["language"] == lang])

    # Summary
    c = Counter((r["language"], r["split"], r["genre"], r["length_tier"]) for r in rows)
    src = Counter((r["language"], r["genre"], r["source"], r["native_lang"]) for r in rows if r["language"] == "en")
    tok = Counter()
    for r in rows:
        tok[(r["language"], r["split"])] += r["max_new_tokens"]
    lines = ["# G-level prompt set", "", f"{len(rows)} rows ({len(rows) // 2} prompts x 2 languages). "
             f"Missing translations: {missing}. Dropped in translation review: {dropped}.", "",
             "## Rows per language, split, genre and tier", "", "| language | split | genre | tier | rows |", "|---|---|---|---|---|"]
    for k, v in sorted(c.items()):
        lines.append("| " + " | ".join(k) + f" | {v} |")
    lines += ["", "## Sources (per prompt pair)", "", "| genre | source | native language | prompts |", "|---|---|---|---|"]
    for (lang, g, s_, nl), v in sorted(src.items()):
        lines.append(f"| {g} | {s_} | {nl} | {v} |")
    lines += ["", "## Upper bound on generated tokens (sum of max_new_tokens)", "", "| language | split | tokens |", "|---|---|---|"]
    for (lang, sp), v in sorted(tok.items()):
        lines.append(f"| {lang} | {sp} | {v:,} |")
    lines += ["", "## Examples", ""]
    shown = set()
    for r in rows:
        key = (r["genre"], r["native_lang"])
        if key in shown or r["split"] == "reserve":
            continue
        pair = [x for x in rows if x["pair_id"] == r["pair_id"]]
        shown.add(key)
        lines.append(f"**{r['genre']}, {r['source']} ({r['native_lang']}-native)**")
        lines.append("")
        for x in pair:
            lines.append(f"- {x['language']}: " + x["messages"][1]["content"].replace("\n", " "))
        lines.append("")
    (out / "G_prompts_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
