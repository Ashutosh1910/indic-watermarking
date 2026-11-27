"""Score trimmed passages with the SynthID-Text verifier (thin wrapper around synthid_wm.SynthIDVerifier).

Needs only the tokenizer and the run's saved config (``<out stem>.config.yaml`` written by
generate.py, which holds the watermark keys), not the model weights. Each passage is re-tokenised from
text (``verify_text``), which is what a detector in the wild sees.

    # watermarked (and control) generations, trimmed by trim.py gen
    python verify_generations.py --config runs/hi.s0.config.yaml --in trimmed/hi.jsonl --out scores/hi.jsonl
    # human passages (trim.py human) with the SAME keys: false-positive rate
    python verify_generations.py --config runs/hi.s0.config.yaml --in trimmed/human_hi.jsonl \\
        --out scores/human_hi.jsonl

Input rows: trim.py output (field "passage"; only accepted passages unless --include-rejected), or
generate.py output directly (field "text"; add --score-ids to also score the sampled token ids).
Outputs: ``--out`` (one row per passage: metadata + decision, z, p_value, score, n_scored, ...) and
``<out stem>.summary.md`` / ``.summary.json``: by language x level x cutoff_kind x cutoff x watermarked,
the share called "watermarked" (TPR for watermarked rows, FPR for controls / human text), mean and
median z, and how many were "insufficient" (fewer than detection.min_scored scorable tokens).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_synthid_to_path, log, md_table, read_jsonl_files, sibling, write_jsonl  # noqa: E402

add_synthid_to_path()
from synthid_wm import SynthIDVerifier, load_config, load_tokenizer  # noqa: E402

META = ("id", "pair_id", "language", "level", "genre", "source", "k", "cutoff_kind", "cutoff", "watermarked",
        "attempt", "split", "experiment", "experiments", "passage_id", "template_id", "sentence_end_rule",
        "n_words", "n_tokens", "ai_char_start", "ai_words", "ai_word_share", "ai_tokens", "ai_token_share",
        "config_sha")
GROUP_KEYS = ("language", "level", "k", "cutoff_kind", "cutoff", "watermarked")


def score_rows(rows: Sequence[dict[str, Any]], verifier: SynthIDVerifier, field: str | None,
               include_rejected: bool, score_ids: bool) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if "accepted" in r and not r["accepted"] and not include_rejected:
            continue
        f = field or ("passage" if "passage" in r else "text")
        text = r.get(f)
        if text is None:
            raise SystemExit(f"row {r.get('id')} has no field {f!r}")
        res = verifier.verify_text(text)
        rec = {k: r.get(k) for k in META if k in r}
        if rec.get("watermarked") is None and (r.get("trim_mode") == "human" or r.get("level") == "human"):
            rec["watermarked"] = "human"
        rec.update(field=f, decision=res.decision, z=res.z, p_value=res.p_value, score=res.score,
                   mean_g=res.mean_g, null_mean=res.null_mean, n_tokens_verified=res.n_tokens,
                   n_scored=res.n_scored, n_masked=res.n_masked, method=res.method, alpha=res.alpha)
        if res.bayesian_posterior is not None:
            rec["bayesian_posterior"] = res.bayesian_posterior
        if score_ids and r.get("token_ids"):
            ri = verifier.verify_ids(r["token_ids"])
            rec.update(ids_decision=ri.decision, ids_z=ri.z, ids_p_value=ri.p_value, ids_n_scored=ri.n_scored)
        out.append(rec)
    return out


def summarize(scored: Sequence[dict[str, Any]], group_keys: Sequence[str]) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for s in scored:
        groups[tuple(s.get(k) for k in group_keys)].append(s)

    def order(k: tuple) -> tuple:
        return tuple((0, v, "") if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, 0, str(v))
                     for v in k)

    out = []
    for key in sorted(groups, key=order):
        ss = groups[key]
        zs = [s["z"] for s in ss if s["n_scored"] > 0 and math.isfinite(s["z"])]
        n_wm = sum(s["decision"] == "watermarked" for s in ss)
        out.append({
            **dict(zip(group_keys, key)), "n": len(ss), "n_watermarked": n_wm,
            "rate_watermarked": n_wm / len(ss),
            "n_insufficient": sum(s["decision"] == "insufficient" for s in ss),
            "mean_z": statistics.fmean(zs) if zs else None,
            "median_z": statistics.median(zs) if zs else None,
            "min_z": min(zs) if zs else None, "max_z": max(zs) if zs else None,
            "mean_n_scored": statistics.fmean(s["n_scored"] for s in ss),
        })
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="the run's saved config (<out stem>.config.yaml of generate.py)")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="trim.py (or generate.py) JSONL file(s)")
    ap.add_argument("--out", required=True, help="scores JSONL; summary files are written next to it")
    ap.add_argument("--tokenizer", help="tokenizer id/dir (default: model.model_id of --config); must match generation")
    ap.add_argument("--field", help="text field to score (default: passage if present, else text)")
    ap.add_argument("--include-rejected", action="store_true", help="also score passages trim.py rejected")
    ap.add_argument("--score-ids", action="store_true", help="generate.py rows: also score the sampled token_ids")
    ap.add_argument("--alpha", type=float, help="override detection.alpha")
    ap.add_argument("--min-scored", type=int, help="override detection.min_scored")
    ap.add_argument("--group-by", default=",".join(GROUP_KEYS))
    args = ap.parse_args(argv)

    cfg = load_config(args.config, {"detection.alpha": args.alpha, "detection.min_scored": args.min_scored})
    tok_name = args.tokenizer or cfg.model.model_id
    tok = load_tokenizer(tok_name)
    verifier = SynthIDVerifier(cfg.watermark, tok, cfg.detection)
    log(f"verifier: {cfg.watermark.depth} keys, ngram_len {cfg.watermark.ngram_len}, tokenizer {tok_name}, "
        f"{cfg.detection.method}, alpha {cfg.detection.alpha}, min_scored {cfg.detection.min_scored}")

    rows = read_jsonl_files(args.inputs)
    try:
        from generate import config_sha
        sha = config_sha(cfg)
        other = {r.get("config_sha") for r in rows if r.get("config_sha")} - {sha}
        if other:
            log(f"warning: rows were generated with config_sha {sorted(other)}, but --config is {sha}")
    except ImportError:  # pragma: no cover
        pass

    scored = score_rows(rows, verifier, args.field, args.include_rejected, args.score_ids)
    out = Path(args.out)
    write_jsonl(out, scored)
    group_keys = [k for k in args.group_by.split(",") if k]
    summary = summarize(scored, group_keys)
    sibling(out, ".summary.json").write_text(json.dumps({
        "config": args.config, "tokenizer": tok_name, "inputs": args.inputs, "n": len(scored),
        "alpha": cfg.detection.alpha, "min_scored": cfg.detection.min_scored, "groups": summary,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    header = group_keys + ["n", "watermarked", "rate", "insufficient", "mean z", "median z", "min z", "max z",
                           "mean scored"]
    table = md_table(header, [[s.get(k) for k in group_keys] + [
        s["n"], s["n_watermarked"], s["rate_watermarked"], s["n_insufficient"], s["mean_z"], s["median_z"],
        s["min_z"], s["max_z"], s["mean_n_scored"]] for s in summary])
    md = (f"# SynthID verification: {out.name}\n\nconfig `{args.config}`, tokenizer `{tok_name}`, alpha "
          f"{cfg.detection.alpha}, min_scored {cfg.detection.min_scored}. `rate` = share called watermarked "
          f"(TPR on watermarked rows, FPR on controls / human text).\n\n{table}\n")
    sibling(out, ".summary.md").write_text(md, encoding="utf-8")
    print(md)
    log(f"wrote {out} ({len(scored)} rows) and {sibling(out, '.summary.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
