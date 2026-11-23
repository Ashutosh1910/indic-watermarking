"""Download every prompt source into raw JSONL files (step 1 of the prompt-set pipeline).

Needs network access to huggingface.co (and raw.githubusercontent.com for PERSUADE 2.0).
Writes only the fields later steps use, one JSON object per line:

    <out>/raw/<source>.jsonl         records with at least: src_id, lang, text (+ source-specific fields)
    <out>/raw/manifest.json          dataset id, revision, license, row counts, errors

Usage:
    python download_sources.py --out ../work            # all sources
    python download_sources.py --out ../work --only aya anudesh
    python download_sources.py --out ../work --skip wildchat

Every source is wrapped in try/except, so one failure (a gated dataset, a renamed repo) never stops
the others; the manifest says what failed and why.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import traceback
from pathlib import Path

from datasets import load_dataset

try:
    from huggingface_hub import HfApi
except Exception:  # pragma: no cover
    HfApi = None


def revision_of(repo_id: str) -> str | None:
    if HfApi is None:
        return None
    try:
        return HfApi().dataset_info(repo_id).sha
    except Exception:
        return None


def first_ok(candidates, **kw):
    """load_dataset on the first (repo, config) pair that works; returns (ds, repo, config)."""
    errors = []
    for repo, config in candidates:
        try:
            ds = load_dataset(repo, config, **kw) if config else load_dataset(repo, **kw)
            return ds, repo, config
        except Exception as e:  # noqa: BLE001
            errors.append(f"{repo}/{config}: {type(e).__name__}: {e}")
    raise RuntimeError(" | ".join(errors))


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def wc(s: str | None) -> int:
    return len((s or "").split())


# ------------------------------------------------------------------------------------- sources


def src_writingprompts(raw: Path) -> dict:
    """Reddit r/WritingPrompts (Fan et al. 2018). Prompts only, deduplicated, with story stats."""
    ds, repo, cfg = first_ok([("euclaise/writingprompts", None)])
    stats: dict[str, dict] = {}
    for split in ds:
        for r in ds[split]:
            p = (r.get("prompt") or "").strip()
            if not p:
                continue
            s = stats.setdefault(p, {"split": split, "n_stories": 0, "story_words": []})
            s["n_stories"] += 1
            s["story_words"].append(wc(r.get("story")))
    rows = []
    for i, (p, s) in enumerate(stats.items()):
        sw = sorted(s["story_words"])
        rows.append({"src_id": f"wp-{i}", "lang": "en", "text": p, "split": s["split"],
                     "n_stories": s["n_stories"], "median_story_words": sw[len(sw) // 2]})
    n = write_jsonl(raw / "writingprompts.jsonl", rows)
    return {"dataset": repo, "license": "MIT", "rows": n}


PERSUADE_TRAIN_GDRIVE_ID = "13phHyDzIsb0MHyJr6q-B-qIa9P2tM135"  # "Training set" link in the repo README


def _persuade_csv_text(raw: Path) -> str:
    """The README says the CSVs live on Google Drive; fetch the training set with gdown."""
    import zipfile

    import gdown  # pip install gdown

    dest = raw / "_persuade_train.bin"
    if not dest.exists() or dest.stat().st_size < 1000:
        gdown.download(id=PERSUADE_TRAIN_GDRIVE_ID, output=str(dest), quiet=False)
    data = dest.read_bytes()
    if data[:2] == b"PK":  # zip archive
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            data = z.read(name)
    return data.decode("utf-8", "replace")


def src_persuade(raw: Path) -> dict:
    """PERSUADE 2.0 (Crossley et al. 2024): the 15 essay prompts (assignment text) + essay length stats."""
    data = _persuade_csv_text(raw)
    csv.field_size_limit(sys.maxsize)
    by_prompt: dict[str, dict] = {}
    for r in csv.DictReader(io.StringIO(data)):
        name = (r.get("prompt_name") or "").strip()
        if not name:
            continue
        p = by_prompt.setdefault(name, {"assignment": (r.get("assignment") or "").strip(),
                                        "task": (r.get("task") or "").strip(),
                                        "has_source_text": bool((r.get("source_text") or "").strip()),
                                        "words": []})
        p["words"].append(wc(r.get("full_text")))
    rows = []
    for i, (name, p) in enumerate(sorted(by_prompt.items())):
        w = sorted(p["words"])
        rows.append({"src_id": f"persuade-{i}", "lang": "en", "text": p["assignment"], "prompt_name": name,
                     "task": p["task"], "has_source_text": p["has_source_text"], "n_essays": len(w),
                     "median_essay_words": w[len(w) // 2]})
    n = write_jsonl(raw / "persuade.jsonl", rows)
    return {"dataset": "PERSUADE 2.0 training set (Google Drive link in github.com/scrosseye/persuade_corpus_2.0)",
            "license": "CC BY 4.0", "rows": n}


def src_no_robots(raw: Path) -> dict:
    ds, repo, cfg = first_ok([("HuggingFaceH4/no_robots", None)])
    rows = []
    for split in ds:
        for r in ds[split]:
            rows.append({"src_id": f"nr-{r['prompt_id']}", "lang": "en", "text": r["prompt"].strip(),
                         "category": r.get("category"), "split": split,
                         "response_words": wc(next((m["content"] for m in r["messages"] if m["role"] == "assistant"), ""))})
    n = write_jsonl(raw / "no_robots.jsonl", rows)
    return {"dataset": repo, "license": "CC BY-NC 4.0", "rows": n, "revision": revision_of(repo)}


def src_dolly(raw: Path) -> dict:
    ds, repo, cfg = first_ok([("databricks/databricks-dolly-15k", None)])
    rows = []
    for i, r in enumerate(ds["train"]):
        rows.append({"src_id": f"dolly-{i}", "lang": "en", "text": r["instruction"].strip(),
                     "context": r.get("context") or "", "category": r.get("category"),
                     "response_words": wc(r.get("response"))})
    n = write_jsonl(raw / "dolly.jsonl", rows)
    return {"dataset": repo, "license": "CC BY-SA 3.0", "rows": n, "revision": revision_of(repo)}


def src_oasst2(raw: Path) -> dict:
    """Root prompts (first user turn of each tree), English and Hindi only."""
    ds, repo, cfg = first_ok([("OpenAssistant/oasst2", None)])
    rows = []
    for split in ds:
        for r in ds[split]:
            if r.get("parent_id") is not None or r.get("role") != "prompter":
                continue
            if r.get("lang") not in ("en", "hi"):
                continue
            if r.get("deleted"):
                continue
            rows.append({"src_id": f"oasst2-{r['message_id']}", "lang": r["lang"], "text": r["text"].strip(),
                         "review_result": r.get("review_result"), "split": split})
    n = write_jsonl(raw / "oasst2.jsonl", rows)
    return {"dataset": repo, "license": "Apache 2.0", "rows": n, "revision": revision_of(repo)}


import re

WRITING_HINT = re.compile(r"\b(write|writing|story|stories|essay|poem|letter|article|blog|speech|narrative|diary|"
                          r"journal|fiction|novel|chapter|scene|monologue|dialogue|tale|fable|memoir|rewrite|"
                          r"continue|opinion|argue|argument|persuasive|op-ed|editorial|review|reflection|describe)\b")
DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def src_wildchat(raw: Path) -> dict:
    """First user turn of each conversation; English, Hindi, or from India. Non-toxic, not redacted."""
    repo = "allenai/WildChat-1M"
    ds = load_dataset(repo, split="train", streaming=True)
    rows, seen, scanned = [], set(), 0
    t0 = time.time()
    for r in ds:
        scanned += 1
        if scanned % 100_000 == 0:
            print(f"  wildchat: scanned {scanned:,}, kept {len(rows):,} ({time.time() - t0:.0f}s)", flush=True)
        if r.get("toxic") or r.get("redacted"):
            continue
        conv = r.get("conversation") or []
        first = next((m for m in conv if m.get("role") == "user"), None)
        if not first:
            continue
        lang = (first.get("language") or r.get("language") or "")
        country = r.get("country") or ""
        if lang not in ("English", "Hindi") and country != "India":
            continue
        text = (first.get("content") or "").strip()
        n = wc(text)
        if n < 5 or n > 600:
            continue
        key = text.lower()
        # Cheap pre-filter so the raw file stays small: keep only requests that mention a writing form
        # or are in Devanagari. The real filter runs later (filter_prompts.py).
        if not (WRITING_HINT.search(key) or DEVANAGARI.search(text)):
            continue
        if key in seen:
            continue
        seen.add(key)
        rows.append({"src_id": f"wildchat-{r['conversation_hash']}", "lang": {"English": "en", "Hindi": "hi"}.get(lang, lang),
                     "text": text, "language": lang, "country": country, "model": r.get("model"),
                     "turns": r.get("turn")})
    n = write_jsonl(raw / "wildchat.jsonl", rows)
    return {"dataset": repo, "license": "ODC-BY", "rows": n, "scanned": scanned, "revision": revision_of(repo)}


def src_aya(raw: Path) -> dict:
    ds, repo, cfg = first_ok([("CohereLabs/aya_dataset", None), ("CohereForAI/aya_dataset", None)])
    rows = []
    for split in ds:
        for i, r in enumerate(ds[split]):
            if r.get("language_code") != "hin":
                continue
            rows.append({"src_id": f"aya-{split}-{i}", "lang": "hi", "text": r["inputs"].strip(),
                         "annotation_type": r.get("annotation_type"), "user_id": r.get("user_id"),
                         "target_words": wc(r.get("targets")), "split": split})
    n = write_jsonl(raw / "aya.jsonl", rows)
    return {"dataset": repo, "license": "Apache 2.0", "rows": n, "revision": revision_of(repo)}


def src_aya_eval_dolly(raw: Path) -> dict:
    """Aya Evaluation Suite, dolly_human_edited, Hindi: Dolly prompts MT'd then post-edited by humans."""
    ds, repo, cfg = first_ok([("CohereLabs/aya_evaluation_suite", "dolly_human_edited"),
                              ("CohereForAI/aya_evaluation_suite", "dolly_human_edited")])
    rows = []
    for split in ds:
        for r in ds[split]:
            if r.get("language") not in ("hin", "hindi", "Hindi"):
                continue
            rows.append({"src_id": f"ayaeval-{r.get('id')}", "lang": "hi", "text": r["inputs"].strip(),
                         "source_id": r.get("source_id"), "split": split})
    n = write_jsonl(raw / "aya_eval_dolly.jsonl", rows)
    return {"dataset": repo, "config": cfg, "license": "Apache 2.0", "rows": n, "revision": revision_of(repo)}


def _first_user(messages) -> str:
    for m in messages or []:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def src_anudesh(raw: Path) -> dict:
    """Anudesh: prompts written by native speakers on the Anudesh platform (IndicInstruct)."""
    tried = []
    for repo, cfg in [("ai4bharat/indic-instruct-data-v0.1", "anudesh"), ("ai4bharat/indic-align", "Anudesh")]:
        try:
            ds = load_dataset(repo, cfg)
        except Exception as e:  # noqa: BLE001
            tried.append(f"{repo}/{cfg}: {type(e).__name__}: {e}")
            continue
        rows = []
        for split in ds:
            for i, r in enumerate(ds[split]):
                text = _first_user(r.get("messages"))
                lang = r.get("lang") or r.get("language") or split
                if not text:
                    # indic-align style: interactions list of [user, assistant] pairs per language
                    inter = r.get("interactions")
                    if inter:
                        text = (inter[0][0] if isinstance(inter[0], (list, tuple)) else "").strip()
                if not text:
                    continue
                rows.append({"src_id": f"anudesh-{split}-{r.get('id', i)}", "lang": "hi" if "hi" in str(lang).lower() else str(lang),
                             "text": text, "split": split})
        n = write_jsonl(raw / "anudesh.jsonl", rows)
        return {"dataset": repo, "config": cfg, "license": "CC BY 4.0 (IndicAlign)", "rows": n,
                "splits": list(ds.keys()), "revision": revision_of(repo), "tried": tried}
    raise RuntimeError(" | ".join(tried))


SOURCES = {
    "persuade": src_persuade,
    "no_robots": src_no_robots,
    "dolly": src_dolly,
    "oasst2": src_oasst2,
    "aya": src_aya,
    "aya_eval_dolly": src_aya_eval_dolly,
    "anudesh": src_anudesh,
    "writingprompts": src_writingprompts,
    "wildchat": src_wildchat,  # biggest; last
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()
    raw = Path(a.out) / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    man_path = raw / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for name, fn in SOURCES.items():
        if a.only and name not in a.only:
            continue
        if name in a.skip:
            continue
        print(f"[{name}] downloading ...", flush=True)
        t0 = time.time()
        try:
            info = fn(raw)
            info.update(ok=True, seconds=round(time.time() - t0, 1))
            print(f"[{name}] ok: {info}", flush=True)
        except Exception as e:  # noqa: BLE001
            info = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
            print(f"[{name}] FAILED: {info['error']}", flush=True)
        manifest[name] = info
        man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: (v.get("rows") if v.get("ok") else "FAILED") for k, v in manifest.items()}, indent=1))


if __name__ == "__main__":
    main()
