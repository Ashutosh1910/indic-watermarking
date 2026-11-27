"""Read a run_study.py dump back for analysis, and write its report.

In Python (pandas needed only for the ``*_df`` helpers)::

    import sys; sys.path.insert(0, "promptset/scripts")
    from load_dump import Dump, rate_table, prefix_z
    d = Dump("dump/study")
    res = d.results_df()                    # one row per scored passage: z, flag, n_tokens, n_words, language, level, ...
    hum = res[res.watermarked == "human"]   # human text: FPR per language x cutoff
    rate_table(hum.to_dict("records"), ["stream", "language", "cutoff_kind", "cutoff"])
    gen = d.generations_df(text=True)       # raw generations (+ token ids with ids=True)
    for t in d.tokens("wm"):                # per-token detector state of every text
        z, n = prefix_z(t, n_tokens=400)    # z of its first 400 tokens

From the shell::

    python promptset/scripts/load_dump.py report dump/study          # -> dump/study/report.md + results/*.csv
    python promptset/scripts/load_dump.py rates dump/study --by language,cutoff_kind,cutoff --where watermarked=human

Tables come from these files of the dump (see promptset/JOB.md):

    gen/<stream>/a<k>.s<i>of<n>.jsonl  raw generations (input row + text, token_ids, seeds, timings, ...)
    trim/<stream>.jsonl                one row per (id, cutoff kind, cutoff): accepted, reject_reason, passage, ...
    trim/<stream>.cleaned.jsonl        text_raw / text_clean per generation and what cleaning removed
    trim/human.<set>.jsonl             the same for human text
    scores/<stream>.jsonl              one row per accepted passage: z, p_value, flag, n_scored, ... (view=passage)
    scores/raw.<stream>.jsonl          the full raw generations, text and sampled ids (view=raw)
    tokens/<stream>.jsonl.gz           per text: token_ids, g_bits, mask (prefix_z recomputes any prefix)
    failures/*.jsonl, *.txt            job, generation, dropped and missing records
    manifest.json, job_runs.jsonl      config_sha, inputs (sha256), environment, stages and their timings
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import iter_jsonl, md_table  # noqa: E402

GEN_FILE = re.compile(r"^a(\d+)\.s(\d+)of(\d+)\.jsonl$")
PASSAGE_KEY = ("stream", "id", "attempt", "watermarked", "cutoff_kind", "cutoff")
HEAVY_GEN = ("messages", "token_ids", "text", "head", "human_passage")
VIEW = {"words": "same_words", "tokens": "same_tokens"}


# --------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def depth_weights(depth: int, method: str) -> list[float]:
    """As synthid_wm.verify.depth_weights: linear 10..1 rescaled to sum to D, or all ones for "mean"."""
    if method == "mean" or depth == 1:
        return [1.0] * depth
    w = [10 - 9 * i / (depth - 1) for i in range(depth)]
    s = sum(w)
    return [x * depth / s for x in w]


def prefix_z(detail: dict[str, Any], n_tokens: int | None = None, method: str | None = None
             ) -> tuple[float, int]:
    """(z, n_scored) of the first ``n_tokens`` tokens of a ``tokens/*.jsonl.gz`` record (all if None).

    Exactly the verifier's statistic for that token prefix. A shorter passage cut from the same text is
    re-tokenised on its own, so its last token or two can differ; its exact z is in scores/."""
    D, ng = int(detail["depth"]), int(detail["ngram_len"])
    m = method or detail.get("method", "weighted_mean")
    w = depth_weights(D, m)
    p0 = float(detail["null_mean"])
    n_pos = len(detail["g_bits"])
    if n_tokens is not None:
        n_pos = max(0, min(n_pos, n_tokens - (ng - 1)))
    tot, n = 0.0, 0
    for bits, keep in zip(detail["g_bits"][:n_pos], detail["mask"][:n_pos]):
        if not keep:
            continue
        n += 1
        tot += sum(w[d] for d in range(D) if (bits >> d) & 1)
    if n == 0:
        return 0.0, 0
    score = tot / (D * n)
    var = p0 * (1 - p0) * sum(x * x for x in w) / (D * D * n)
    return (score - p0) / math.sqrt(var), n


def g_matrix(detail: dict[str, Any]) -> list[list[int]]:
    """Unpack g_bits to a [positions x depth] 0/1 matrix."""
    D = int(detail["depth"])
    return [[(b >> d) & 1 for d in range(D)] for b in detail["g_bits"]]


def rate_table(rows: Sequence[dict[str, Any]], by: Sequence[str], z_threshold: float | None = None,
               min_scored: int | None = None) -> list[dict[str, Any]]:
    """Share of texts flagged per group, with a 95 % Wilson interval and z quantiles.

    Flag: the row's ``flag`` (z > threshold with enough scored tokens), or recomputed from ``z`` /
    ``n_scored`` when ``z_threshold`` (and optionally ``min_scored``) is given."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in by)].append(r)

    def flag(r: dict[str, Any]) -> bool:
        if z_threshold is None and "flag" in r:
            return bool(r["flag"])
        zt = 3.0 if z_threshold is None else z_threshold
        ms = min_scored if min_scored is not None else 0
        return r["n_scored"] >= ms and r["z"] > zt and r.get("decision") != "insufficient"

    def order(k: tuple) -> tuple:
        return tuple((0, v, "") if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, 0, str(v)) for v in k)

    out = []
    for key in sorted(groups, key=order):
        g = groups[key]
        zs = sorted(r["z"] for r in g if r.get("n_scored") and math.isfinite(r["z"]))
        k = sum(flag(r) for r in g)
        lo, hi = wilson(k, len(g))
        out.append({**dict(zip(by, key)), "n": len(g), "flagged": k, "rate": k / len(g), "ci_lo": lo, "ci_hi": hi,
                    "mean_z": statistics.fmean(zs) if zs else None, "median_z": statistics.median(zs) if zs else None,
                    "p95_z": zs[min(len(zs) - 1, int(0.95 * len(zs)))] if zs else None,
                    "max_z": zs[-1] if zs else None,
                    "mean_n_tokens": _mean(r.get("n_tokens") or r.get("n_tokens_verified") for r in g),
                    "mean_n_words": _mean(r.get("n_words") for r in g),
                    "insufficient": sum(r.get("decision") == "insufficient" for r in g)})
    return out


def _mean(vals: Iterable[Any]) -> float | None:
    v = [x for x in vals if isinstance(x, (int, float))]
    return statistics.fmean(v) if v else None


# --------------------------------------------------------------------------- the dump


def _read(path: Path) -> Iterator[dict[str, Any]]:
    """JSONL rows; a partial last line (a job still writing, or killed) is skipped."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


class Dump:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(self.path)

    # ---- metadata
    def manifest(self) -> dict[str, Any]:
        p = self.path / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def job_runs(self) -> list[dict[str, Any]]:
        p = self.path / "job_runs.jsonl"
        return list(_read(p)) if p.exists() else []

    def streams(self) -> list[str]:
        d = self.path / "gen"
        return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []

    def human_sets(self) -> list[str]:
        return sorted(p.name[len("human."):-len(".jsonl")] for p in (self.path / "trim").glob("human.*.jsonl")
                      if p.name.count(".") == 2)

    # ---- rows
    def generations(self, stream: str | None = None, text: bool = False, ids: bool = False
                    ) -> Iterator[dict[str, Any]]:
        """Raw generation rows. Heavy fields (messages, token_ids, text, head, human_passage) are dropped
        unless ``text`` (keeps text) / ``ids`` (keeps token_ids) is set."""
        for s in [stream] if stream else self.streams():
            for p in sorted((self.path / "gen" / s).glob("a*.jsonl")):
                m = GEN_FILE.match(p.name)
                if not m:
                    continue
                for r in _read(p):
                    keep = {k: v for k, v in r.items()
                            if k not in HEAVY_GEN or (text and k == "text") or (ids and k == "token_ids")}
                    keep.update(stream=s, shard_file=p.name)
                    yield keep

    def passages(self, stream: str | None = None, text: bool = False) -> Iterator[dict[str, Any]]:
        """Trimmed passages (accepted and rejected) of AI streams and human sets; ``stream`` names either
        (human sets are ``human.<set>``)."""
        names = [stream] if stream else [p.name[:-len(".jsonl")] for p in sorted((self.path / "trim").glob("*.jsonl"))
                                         if p.name.count(".") <= 2 and not p.name.endswith(
                                             (".cleaned.jsonl", ".summary.jsonl"))]
        for n in names:
            p = self.path / "trim" / f"{n}.jsonl"
            if not p.exists() or ".cleaned" in n or ".regen_ids" in n:
                continue
            for r in _read(p):
                if not text:
                    r.pop("passage", None)
                r["stream"] = n
                r["view"] = VIEW.get(r.get("cutoff_kind"))
                if r.get("watermarked") is None and (r.get("trim_mode") == "human" or r.get("level") == "human"):
                    r["watermarked"] = "human"   # as verify_generations labels them
                yield r

    def cleaned(self, stream: str) -> Iterator[dict[str, Any]]:
        yield from _read(self.path / "trim" / f"{stream}.cleaned.jsonl")

    def scores(self, view: str | None = "passage") -> Iterator[dict[str, Any]]:
        """Score rows; view "passage" (trimmed passages, AI and human), "raw" (full generations) or None (both)."""
        for p in sorted((self.path / "scores").glob("*.jsonl")):
            is_raw = p.name.startswith("raw.")
            if view == "passage" and is_raw or view == "raw" and not is_raw:
                continue
            for r in _read(p):
                r.setdefault("stream", p.name[len("raw."):-6] if is_raw else p.name[:-6])
                r.setdefault("view", "raw" if is_raw else "passage")
                if r.get("cutoff_kind"):
                    r["experiment_view"] = VIEW.get(r["cutoff_kind"])
                yield r

    def results(self, text: bool = False) -> list[dict[str, Any]]:
        """Passage scores joined with every field of their trimmed passage (one row per scored passage)."""
        idx = {tuple(_key(r)): r for r in self.passages(text=text)}
        out = []
        for s in self.scores("passage"):
            p = idx.get(tuple(_key(s)), {})
            out.append({**p, **s})
        return out

    def tokens(self, stream: str) -> Iterator[dict[str, Any]]:
        p = self.path / "tokens" / f"{stream}.jsonl.gz"
        if p.exists():
            yield from _read(p)

    def failures(self) -> dict[str, list[dict[str, Any]]]:
        f = self.path / "failures"
        out: dict[str, list[dict[str, Any]]] = {"job": [], "generate": [], "dropped": [], "missing": []}
        if not f.is_dir():
            return out
        for p in sorted(f.iterdir()):
            if p.name == "job.jsonl":
                out["job"] += list(_read(p))
            elif p.name.startswith("gen."):
                out["generate"] += [{**r, "stream": p.name[4:-6]} for r in _read(p)]
            elif p.name.startswith("dropped."):
                out["dropped"] += list(_read(p))
            elif p.name.startswith("missing."):
                s = p.name[len("missing."):-4]
                out["missing"] += [{"stream": s, "id": l.strip()} for l in p.read_text(encoding="utf-8").splitlines()
                                   if l.strip()]
        return out

    # ---- pandas
    def _df(self, rows: Iterable[dict[str, Any]]):
        import pandas as pd
        return pd.DataFrame(list(rows))

    def generations_df(self, stream: str | None = None, text: bool = False, ids: bool = False):
        return self._df(self.generations(stream, text, ids))

    def passages_df(self, stream: str | None = None, text: bool = False):
        return self._df(self.passages(stream, text))

    def scores_df(self, view: str | None = "passage"):
        return self._df(self.scores(view))

    def results_df(self, text: bool = False):
        return self._df(self.results(text))


def _key(r: dict[str, Any]) -> list[Any]:
    return [r.get("stream"), str(r.get("id")), r.get("attempt"), r.get("watermarked"), r.get("cutoff_kind"),
            r.get("cutoff")]


# --------------------------------------------------------------------------- report


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                        for k, v in r.items()})


def _fmt_rate(r: dict[str, Any]) -> str:
    if r["ci_lo"] is None:
        return ""
    return f"{r['rate']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def rate_md(rows: Sequence[dict[str, Any]], by: Sequence[str]) -> str:
    return md_table(list(by) + ["n", "flagged", "rate [95% CI]", "mean z", "median z", "p95 z", "mean tokens",
                                "mean words"],
                    [[r[k] for k in by] + [r["n"], r["flagged"], _fmt_rate(r), r["mean_z"], r["median_z"], r["p95_z"],
                                           r["mean_n_tokens"], r["mean_n_words"]] for r in rows])


def report(dump: Dump, z_threshold: float) -> str:
    man = dump.manifest()
    res = dump.results()
    out_dir = dump.path / "results"
    lines = [f"# Study report: {dump.path.name}", ""]
    if man:
        lines += [f"Status **{man.get('status')}**, finished {man.get('finished_utc')}. Model `{man.get('model_id')}`, "
                  f"config_sha `{man.get('config_sha')}`, tokenizer `{man.get('tokenizer')}`, GPUs {man.get('gpus_used')}. "
                  f"Flag: z > {z_threshold} with at least min_scored scored tokens.", ""]
        st = man.get("stages") or []
        lines += ["## Stages (last run)", "", md_table(["stage", "status", "seconds", "detail"], [
            [s["stage"], s["status"], s["seconds"],
             json.dumps({k: v for k, v in s.items() if k not in ("stage", "status", "seconds")}, ensure_ascii=False)[:160]]
            for s in st]), ""]

    # generation
    gen_groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for g in dump.generations():
        gen_groups[(g["stream"], g.get("language"), g.get("level"), g.get("attempt"))].append(g)
    gen_rows = []
    for (s, lang, lev, a), g in sorted(gen_groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        secs = {}
        for r in g:   # a batch's wall time is on each of its rows
            secs[(r["shard_file"], r.get("batch_seed"), r.get("gen_time_utc"))] = r.get("seconds") or 0
        toks = sum(r.get("n_new_tokens") or 0 for r in g)
        gen_rows.append({"stream": s, "language": lang, "level": lev, "attempt": a, "rows": len(g),
                         "share_length_stop": sum(r.get("finish_reason") == "length" for r in g) / len(g),
                         "had_think": sum(bool(r.get("had_think")) for r in g),
                         "mean_new_tokens": toks / len(g), "new_tokens": toks,
                         "gpu_seconds": round(sum(secs.values()), 1)})
    write_csv(out_dir / "generation.csv", gen_rows)
    if gen_rows:
        lines += ["## Generation", "", md_table(list(gen_rows[0]), [list(r.values()) for r in gen_rows]), ""]

    # trimming acceptance
    acc: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    for p in dump.passages():
        k = (p["stream"], p.get("language"), p.get("level"), p.get("k"), p.get("cutoff_kind"), p.get("cutoff"))
        acc[k][0] += bool(p.get("accepted"))
        acc[k][1] += 1
    acc_rows = [{"stream": k[0], "language": k[1], "level": k[2], "k": k[3], "cutoff_kind": k[4], "cutoff": k[5],
                 "accepted": v[0], "total": v[1], "share": v[0] / v[1]}
                for k, v in sorted(acc.items(), key=lambda kv: tuple(str(x) for x in kv[0][:5]) + (kv[0][5] or 0,))]
    write_csv(out_dir / "acceptance.csv", acc_rows)

    by = ["stream", "language", "level", "k", "cutoff_kind", "cutoff"]
    rates = rate_table(res, by, z_threshold)
    write_csv(out_dir / "detection_rates.csv", rates)
    write_csv(out_dir / "detection_rates_by_view.csv",
              rate_table(res, ["watermarked", "language", "cutoff_kind", "cutoff"], z_threshold))
    raw = list(dump.scores("raw"))
    raw_rates = rate_table(raw, ["stream", "language", "level"], z_threshold)
    write_csv(out_dir / "detection_rates_raw.csv", raw_rates)

    wm = [r for r in rates if not str(r["stream"]).startswith("human.") and r["stream"] != "ctrl"]
    ctrl = [r for r in rates if r["stream"] == "ctrl"]
    hum = rate_table([r for r in res if r.get("watermarked") == "human"],
                     ["stream", "language", "cutoff_kind", "cutoff"], z_threshold)
    write_csv(out_dir / "fpr_human.csv", hum)
    lines += ["## Human text: false-positive rate (same keys, same trimming)", "",
              "Per language and length. `rate` = share flagged.", "",
              rate_md(hum, ["stream", "language", "cutoff_kind", "cutoff"]) if hum else "(no human scores)", ""]
    lines += ["## Watermarked AI text: detection rate", "", rate_md(wm, by) if wm else "(no scores)", ""]
    if ctrl:
        lines += ["## Unwatermarked AI text (control): false-positive rate", "", rate_md(ctrl, by), ""]
    if raw_rates:
        lines += ["## Full raw generations (before trimming)", "", rate_md(raw_rates, ["stream", "language", "level"]), ""]

    acc_md = [r for r in acc_rows]
    if acc_md:
        lines += ["## Trimming acceptance", "", md_table(list(acc_md[0]), [list(r.values()) for r in acc_md]), ""]

    f = dump.failures()
    lines += ["## Failures", "", md_table(["kind", "n", "where"], [
        ["job stage failures", len(f["job"]), "failures/job.jsonl"],
        ["failed generation batches", len(f["generate"]), "failures/gen.<stream>.jsonl"],
        ["ids whose largest cutoff failed after the last attempt", len(f["dropped"]), "failures/dropped.<stream>.jsonl"],
        ["selected prompt rows never generated", len(f["missing"]), "failures/missing.<stream>.txt"]]), ""]
    text = "\n".join(lines) + "\n"
    (dump.path / "report.md").write_text(text, encoding="utf-8")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="write <dump>/report.md and <dump>/results/*.csv")
    r.add_argument("dump")
    r.add_argument("--z-threshold", type=float, default=3.0)
    q = sub.add_parser("rates", help="print a flag-rate table of the passage scores")
    q.add_argument("dump")
    q.add_argument("--by", default="stream,language,level,cutoff_kind,cutoff")
    q.add_argument("--where", action="append", default=[], metavar="KEY=VALUE")
    q.add_argument("--z-threshold", type=float, default=None, help="recompute flags at this z (default: as scored)")
    args = ap.parse_args(argv)
    d = Dump(args.dump)
    if args.cmd == "report":
        report(d, args.z_threshold)
        print(f"wrote {d.path / 'report.md'} and {d.path / 'results'}/*.csv")
        return 0
    rows = d.results()
    for w in args.where:
        k, _, v = w.partition("=")
        rows = [x for x in rows if str(x.get(k)) == v]
    by = [k for k in args.by.split(",") if k]
    print(rate_md(rate_table(rows, by, args.z_threshold), by))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
