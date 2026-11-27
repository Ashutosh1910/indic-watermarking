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

Job options (used by run_study.py):

* ``--z-threshold Z`` sets alpha = P(N(0,1) > Z), so "watermarked" means z > Z (the study uses 3). Every
  row also gets ``z_threshold`` and ``flag`` (z > Z and enough scored tokens).
* ``--tag KEY=VALUE`` adds a constant column to every row (e.g. ``stream=wm``, ``view=passage``).
* ``--details OUT.jsonl.gz`` writes the per-token detector state (token ids, packed g-values, repetition
  mask; ``SynthIDVerifier.token_details``) once per text: for each (id, attempt, watermarked) the LONGEST
  scored passage, of which every shorter cutoff is a prefix. ``load_dump.prefix_z`` recomputes any prefix.
* ``--workers N`` scores in N processes (CPU).
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import multiprocessing as mp
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


def z_to_alpha(z: float) -> float:
    """One-sided tail P(N(0,1) > z): with this alpha, p < alpha <=> z > z_threshold."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def parse_tags(specs: Sequence[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for spec in specs or []:
        k, sep, v = spec.partition("=")
        if not sep or not k:
            raise SystemExit(f"--tag expects KEY=VALUE, got {spec!r}")
        out[k] = v
    return out


def _text_field(r: dict[str, Any], field: str | None) -> str:
    f = field or ("passage" if "passage" in r else "text")
    if r.get(f) is None:
        raise SystemExit(f"row {r.get('id')} has no field {f!r}")
    return f


def score_one(r: dict[str, Any], verifier: SynthIDVerifier, field: str | None, score_ids: bool,
              z_threshold: float | None = None, tags: dict[str, Any] | None = None) -> dict[str, Any]:
    f = _text_field(r, field)
    res = verifier.verify_text(r[f])
    rec = {k: r.get(k) for k in META if k in r}
    if rec.get("watermarked") is None and (r.get("trim_mode") == "human" or r.get("level") == "human"):
        rec["watermarked"] = "human"
    rec.update(field=f, decision=res.decision, z=res.z, p_value=res.p_value, score=res.score,
               mean_g=res.mean_g, null_mean=res.null_mean, n_tokens_verified=res.n_tokens,
               n_scored=res.n_scored, n_masked=res.n_masked, method=res.method, alpha=res.alpha,
               n_chars=len(r[f]))
    if z_threshold is not None:
        rec.update(z_threshold=z_threshold,
                   flag=bool(res.n_scored >= verifier.det.min_scored and res.z > z_threshold))
    if res.bayesian_posterior is not None:
        rec["bayesian_posterior"] = res.bayesian_posterior
    if score_ids and r.get("token_ids"):
        ri = verifier.verify_ids(r["token_ids"])
        rec.update(ids_decision=ri.decision, ids_z=ri.z, ids_p_value=ri.p_value, ids_n_scored=ri.n_scored,
                   ids_n_tokens=ri.n_tokens)
    if tags:
        rec.update(tags)
    return rec


def wanted(r: dict[str, Any], include_rejected: bool) -> bool:
    return include_rejected or "accepted" not in r or bool(r["accepted"])


def score_rows(rows: Sequence[dict[str, Any]], verifier: SynthIDVerifier, field: str | None,
               include_rejected: bool, score_ids: bool, z_threshold: float | None = None,
               tags: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [score_one(r, verifier, field, score_ids, z_threshold, tags) for r in rows if wanted(r, include_rejected)]


def detail_groups(rows: Sequence[dict[str, Any]], field: str | None, include_rejected: bool
                  ) -> list[dict[str, Any]]:
    """For each (id, attempt, watermarked): the row with the longest text (all shorter cutoffs are prefixes)."""
    best: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        if not wanted(r, include_rejected):
            continue
        key = (str(r.get("id")), r.get("attempt"), r.get("watermarked"))
        f = _text_field(r, field)
        if key not in best or len(r[f]) > len(best[key][_text_field(best[key], field)]):
            best[key] = r
    return list(best.values())


def detail_record(r: dict[str, Any], verifier: SynthIDVerifier, field: str | None,
                  tags: dict[str, Any] | None = None) -> dict[str, Any]:
    f = _text_field(r, field)
    ids = verifier.tokenizer(r[f], add_special_tokens=False)["input_ids"]
    rec = {"id": r.get("id"), "attempt": r.get("attempt"), "watermarked": r.get("watermarked"),
           "language": r.get("language"), "level": r.get("level"), "cutoff_kind": r.get("cutoff_kind"),
           "cutoff": r.get("cutoff"), "n_chars": len(r[f])}
    if rec["watermarked"] is None and (r.get("trim_mode") == "human" or r.get("level") == "human"):
        rec["watermarked"] = "human"
    rec.update(verifier.token_details(ids))
    if tags:
        rec.update(tags)
    return rec


# multiprocessing: one verifier per worker process
_W: dict[str, Any] = {}


def _init_worker(config: str, tokenizer: str, overrides: dict[str, Any]) -> None:
    import torch
    torch.set_num_threads(1)
    cfg = load_config(config, overrides)
    _W["v"] = SynthIDVerifier(cfg.watermark, load_tokenizer(tokenizer), cfg.detection)


def _work(job: tuple) -> dict[str, Any]:
    kind, r, field, score_ids, z_threshold, tags = job
    if kind == "score":
        return score_one(r, _W["v"], field, score_ids, z_threshold, tags)
    return detail_record(r, _W["v"], field, tags)


def run_parallel(jobs: list[tuple], workers: int, config: str, tokenizer: str, overrides: dict[str, Any],
                 verifier: SynthIDVerifier) -> list[dict[str, Any]]:
    if workers <= 1 or len(jobs) < 2 * workers:
        _W["v"] = verifier
        return [_work(j) for j in jobs]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(config, tokenizer, overrides)) as pool:
        return list(pool.imap(_work, jobs, chunksize=max(1, min(64, len(jobs) // (workers * 8)))))


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
    ap.add_argument("--z-threshold", type=float, help="call a text watermarked iff z > Z (sets alpha; overrides --alpha)")
    ap.add_argument("--tag", action="append", metavar="KEY=VALUE", help="constant column added to every row")
    ap.add_argument("--details", help="per-token detector state, one record per text (JSONL, .gz to compress)")
    ap.add_argument("--workers", type=int, default=1, help="scoring processes (default 1)")
    args = ap.parse_args(argv)

    alpha = z_to_alpha(args.z_threshold) if args.z_threshold is not None else args.alpha
    overrides = {"detection.alpha": alpha, "detection.min_scored": args.min_scored}
    cfg = load_config(args.config, overrides)
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

    tags = parse_tags(args.tag)
    jobs = [("score", r, args.field, args.score_ids, args.z_threshold, tags)
            for r in rows if wanted(r, args.include_rejected)]
    n_score = len(jobs)
    if args.details:
        jobs += [("details", r, args.field, False, None, tags)
                 for r in detail_groups(rows, args.field, args.include_rejected)]
    results = run_parallel(jobs, args.workers, args.config, tok_name, overrides, verifier)
    scored, details = results[:n_score], results[n_score:]
    out = Path(args.out)
    write_jsonl(out, scored)
    if args.details:
        dp = Path(args.details)
        dp.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if dp.suffix == ".gz" else open
        with opener(dp, "wt", encoding="utf-8") as fh:
            for d in details:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
        log(f"wrote {dp} ({len(details)} texts)")
    group_keys = [k for k in args.group_by.split(",") if k]
    summary = summarize(scored, group_keys)
    sibling(out, ".summary.json").write_text(json.dumps({
        "config": args.config, "tokenizer": tok_name, "inputs": args.inputs, "n": len(scored),
        "alpha": cfg.detection.alpha, "z_threshold": args.z_threshold, "min_scored": cfg.detection.min_scored,
        "method": cfg.detection.method, "null_mean": verifier.p0, "tags": tags, "groups": summary,
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
