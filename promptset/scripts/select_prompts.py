"""Pick the prompt set from checked candidates, balanced by genre and native language (step 4).

Inputs (under --work):
    candidates/<source>.jsonl     from filter_prompts.py
    labels/*.jsonl                LLM check (LABELING.md): cid, keep, genre, reason
Outputs:
    selected.jsonl                one row per prompt pair: pair_id, native_lang, genre, source, split, tier, text
    to_translate.jsonl            id, src, text  -> translate_indictrans2.py
    selection_report.json         counts by genre x native language x source x split x tier

Design: every prompt is used in both languages (its native one and a translation), so English and Hindi
get exactly the same prompts, genres and length tiers. In each genre about half the prompts are
Hindi-native (as far as the Hindi pool allows) and the rest English-native, so "translated vs native"
can be checked too and no language owns a genre.

    python select_prompts.py --work ../work --config ../config/promptset.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def cid_of(r: dict) -> str:
    return f"{r['source']}:{r['src_id']}"


def stable_shuffle(items: list, seed: str) -> list:
    rnd = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    items = list(items)
    rnd.shuffle(items)
    return items


def weighted_take(pools: dict[str, list], weights: dict[str, float], n: int) -> list:
    """Take n items across sources in proportion to weights (largest remainder), skipping empty pools."""
    taken: list = []
    pools = {s: list(p) for s, p in pools.items() if p}
    credit = Counter()
    while len(taken) < n and pools:
        live = {s: weights.get(s, 1) for s in pools if weights.get(s, 1) > 0} or {s: 1 for s in pools}  # unlisted source: weight 1
        tot = sum(live.values())
        for s, w in live.items():
            credit[s] += w / tot
        s = max(live, key=lambda k: credit[k])
        credit[s] -= 1
        taken.append(pools[s].pop(0))
        if not pools[s]:
            del pools[s]
    return taken


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    work = Path(a.work)
    cfg = yaml.safe_load(open(a.config))
    seed = str(cfg["seed"])

    cands = {}
    for p in sorted((work / "candidates").glob("*.jsonl")):
        for l in p.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                cands[cid_of(r)] = r
    labels = {}
    for p in sorted((work / "labels").glob("*.jsonl")):
        for l in p.open(encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                labels[r["cid"]] = r
    print(f"{len(cands):,} candidates, {len(labels):,} labels")

    # pools[genre][lang][source] -> list of candidate rows (labelled keep)
    pools: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen_keys: set = set()
    for cid, lab in sorted(labels.items()):
        if not lab.get("keep") or cid not in cands:
            continue
        # Near-duplicates (same first 12 words, e.g. one WildChat user's repeated template): keep the first.
        nk = (cands[cid]["lang"], " ".join(re.findall(r"\w+", cands[cid]["text"].lower())[:12]))
        if nk in seen_keys:
            continue
        seen_keys.add(nk)
        r = dict(cands[cid], genre=lab["genre"], label_reason=lab.get("reason"))
        if r["genre"] not in cfg["genres"]:
            continue
        if r["source"] == "writingprompts" and r["genre"] != "story":
            continue  # WP rows are rendered with the story-premise template
        pools[r["genre"]][r["lang"]][r["source"]].append(r)
    for g in pools:
        for lang in pools[g]:
            for s in pools[g][lang]:
                pools[g][lang][s] = stable_shuffle(pools[g][lang][s], f"{seed}/{g}/{lang}/{s}")

    n_main, n_res = cfg["per_genre"]["main"], cfg["per_genre"]["reserve"]
    total = n_main + n_res
    tiers = list(cfg["tiers"])
    selected, report = [], {}
    for g in cfg["genres"]:
        hi_pool = pools[g]["hi"]
        hi_avail = sum(len(v) for v in hi_pool.values())
        n_hi = min(hi_avail, round(total * cfg["hindi_native_share"]))
        hi_rows = weighted_take(hi_pool, {}, n_hi)  # Hindi sources: equal weights
        en_w = cfg["english_source_weights"].get(g, {})
        en_rows = weighted_take(pools[g]["en"], en_w, total - len(hi_rows))
        if len(en_rows) + len(hi_rows) < total:
            print(f"WARNING genre {g}: only {len(en_rows) + len(hi_rows)} of {total} prompts available")
        # Split each native group main/reserve in proportion, then tiers alternate within each stratum.
        # Fill "main" first (up to n_main), split between native languages in proportion; the rest is reserve.
        avail = len(hi_rows) + len(en_rows)
        main_total = min(n_main, avail)
        hi_main = round(main_total * len(hi_rows) / avail) if avail else 0
        quota = {"hi": hi_main, "en": main_total - hi_main}
        rows = []
        for lang, grp in (("hi", hi_rows), ("en", en_rows)):
            for i, r in enumerate(grp):
                split = "main" if i < quota[lang] else "reserve"
                rows.append((lang, split, r))
        strata = defaultdict(list)
        for lang, split, r in rows:
            strata[(lang, split)].append(r)
        picked = []
        for (lang, split), grp in sorted(strata.items()):
            for i, r in enumerate(grp):
                picked.append({"lang": lang, "split": split, "tier": tiers[i % len(tiers)], "r": r, "i": i})
        # Pilot: per tier, alternate Hindi-native and English-native main rows.
        for tier in tiers:
            queues = {lang: [p for p in picked if p["split"] == "main" and p["tier"] == tier and p["lang"] == lang]
                      for lang in ("hi", "en")}
            k, turn = 0, ["hi", "en"]
            while k < cfg["pilot_per_genre_tier"] and any(queues.values()):
                lang = turn[k % 2] if queues[turn[k % 2]] else turn[(k + 1) % 2]
                queues[lang].pop(0)["split"] = "pilot"
                k += 1
        for p_ in picked:
            r = p_["r"]
            selected.append({
                "pair_id": cid_of(r), "native_lang": p_["lang"], "genre": g, "source": r["source"],
                "src_id": r["src_id"], "split": p_["split"], "tier": p_["tier"], "text": r["text"],
                "wp_tags": r.get("wp_tags"), "label_reason": r.get("label_reason"),
                "source_meta": {k: r[k] for k in ("category", "annotation_type", "prompt_name", "country",
                                                   "median_story_words", "median_essay_words") if k in r},
            })
        report[g] = {"hindi_native_available": hi_avail,
                     "english_native_available": {s: len(v) for s, v in pools[g]["en"].items()},
                     "hindi_native_by_source": dict(Counter(r["source"] for r in hi_rows)),
                     "english_native_by_source": dict(Counter(r["source"] for r in en_rows))}

    with (work / "selected.jsonl").open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (work / "to_translate.jsonl").open("w", encoding="utf-8") as f:
        for r in selected:
            f.write(json.dumps({"id": r["pair_id"], "src": r["native_lang"], "text": r["text"]}, ensure_ascii=False) + "\n")
    cnt = Counter((r["genre"], r["native_lang"], r["split"], r["tier"]) for r in selected)
    report["_counts"] = {"/".join(k): v for k, v in sorted(cnt.items())}
    report["_total"] = len(selected)
    (work / "selection_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report["_counts"], indent=1))
    print(f"{len(selected)} prompt pairs selected")


if __name__ == "__main__":
    main()
