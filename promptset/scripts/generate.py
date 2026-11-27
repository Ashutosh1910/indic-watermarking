"""Generate (SynthID-watermarked) answers for prompt-set JSONL rows with Hugging Face ``generate``.

One output row per prompt row: every input field is passed through untouched (``messages`` too), plus

    text            decoded answer (skip_special_tokens; any <think>...</think> block removed)
    token_ids       sampled ids, up to and including the first EOS
    n_new_tokens    len(token_ids)
    finish_reason   "eos" | "length"
    had_think       true if a <think> block was stripped from the text
    attempt         --attempt (0 for the first try; regenerations use 1, 2, ...)
    watermarked     false with --no-watermark or a row's "watermark": false
    seed_effective  per-row seed actually used (row generation.seed, re-hashed with the attempt if > 0)
    batch_seed      torch.manual_seed value of the batch the row was generated in
    batch_size, prompt_tokens, max_new_tokens_effective, min_new_tokens_effective
    model_id, config_sha (sha256 of the watermark + sampling settings), seconds (wall time of the
    row's batch), gen_time_utc

The watermark is synthid_wm's ``PortableSynthIDConfig`` (transformers' SynthID-Text processor with a
device-independent sampling table), passed as ``model.generate(watermarking_config=...)``, so it runs
after temperature / top-k / top-p. SynthID in transformers needs HF ``generate``; vLLM and other
servers do not apply it.

Usage (from the repo root)::

    python promptset/scripts/generate.py --prompts prompts.jsonl --out runs/hi_pilot.jsonl \\
        --config promptset/config/generation.yaml --filter language=hi --filter split=pilot
    # several GPUs: one process per GPU
    CUDA_VISIBLE_DEVICES=1 python promptset/scripts/generate.py ... --shard 1/4 --out runs/hi.s1.jsonl
    # regenerate failures reported by trim.py
    python promptset/scripts/generate.py ... --only-ids trimmed/hi.regen_ids.txt --attempt 1

Resumable: rows whose (id, attempt, watermarked) is already in --out are skipped, so re-running the
same command after a crash continues where it stopped. The effective config is written next to --out
as ``<out stem>.config.yaml`` (loadable by ``synthid_wm.load_config``; it contains the watermark keys).

Reproducibility: every batch is seeded with ``torch.manual_seed(hash(row seeds in the batch, attempt))``.
Sampling is batched, so reproducing a row exactly needs the same batch composition (same rows, same
--batch-size / --token-budget, no OOM split) and the same hardware/software; bf16 kernels may also
differ across GPU models and library versions.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    add_synthid_to_path,
    dumps,
    log,
    parse_filters,
    read_id_list,
    read_jsonl_files,
    row_matches,
    sibling,
    stable_hash,
)

add_synthid_to_path()
from synthid_wm import Config, PortableSynthIDConfig, load_model, load_tokenizer  # noqa: E402
from synthid_wm.config import _build as _build_synthid_config  # noqa: E402  (same validation as load_config)
from synthid_wm.watermark import eos_ids  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "generation.yaml"


# --------------------------------------------------------------------------- config


@dataclass
class BatchingConfig:
    batch_size: int = 32        # hard cap on rows per generate() call
    token_budget: int = 160_000  # batch_size * (longest prompt + max_new_tokens) <= token_budget


def load_run_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> tuple[Config, BatchingConfig]:
    """Read a run YAML: the four synthid sections (validated by synthid_wm) plus ``batching``.

    ``overrides`` maps dotted keys (``model.model_id``, ``batching.batch_size``) to values; ``None``
    values are ignored, as in ``synthid_wm.load_config``.
    """
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, name = dotted.partition(".")
        raw[section] = dict(raw.get(section) or {})
        raw[section][name] = value
    batching_raw = raw.pop("batching", None) or {}
    known = {f.name for f in dataclasses.fields(BatchingConfig)}
    bad = set(batching_raw) - known
    if bad:
        raise ValueError(f"unknown field(s) in 'batching': {sorted(bad)}")
    batching = BatchingConfig(**batching_raw)
    if batching.batch_size < 1 or batching.token_budget < 1:
        raise ValueError("batching.batch_size and batching.token_budget must be >= 1")
    return _build_synthid_config(raw), batching


def config_sha(cfg: Config) -> str:
    """Hash of everything that changes the sampled distribution for a given prompt: the watermark block
    and the sampling settings. ``max_new_tokens`` and ``seed`` are excluded (they are per row)."""
    gen = {k: v for k, v in dataclasses.asdict(cfg.generation).items() if k not in ("max_new_tokens", "seed")}
    payload = {"watermark": dataclasses.asdict(cfg.watermark), "generation": gen}
    return f"{stable_hash(payload, bits=64):016x}"


def dump_config(cfg: Config) -> str:
    """The synthid part of the effective config as YAML (loadable by synthid_wm.load_config)."""
    head = ("# Effective config written by promptset/scripts/generate.py. Loadable by synthid_wm.load_config\n"
            "# and `python -m demo verify --config`. Contains the watermark keys: keep it private.\n"
            "# generation.max_new_tokens / seed are fallbacks: each row's own values were used.\n")
    return head + yaml.safe_dump(cfg.to_dict(), sort_keys=False, allow_unicode=True, default_flow_style=None, width=1000)


# --------------------------------------------------------------------------- row selection


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    """``"i/n"`` with 0 <= i < n (0-based)."""
    if not spec:
        return None
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"--shard expects i/n, got {spec!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 0 <= i < n:
        raise ValueError(f"--shard {spec}: need 0 <= i < n (shards are 0-based)")
    return i, n


def in_shard(row_id: str, shard: tuple[int, int] | None) -> bool:
    """Shards partition rows by a stable hash of the id, so they do not depend on filters or file order."""
    return shard is None or stable_hash("shard", row_id) % shard[1] == shard[0]


def select_rows(rows: Sequence[dict[str, Any]], filters: dict[str, set[str]], only_ids: set[str] | None,
                shard: tuple[int, int] | None, limit: int | None) -> list[dict[str, Any]]:
    """Filter -> only-ids -> shard -> de-duplicate ids (first wins) -> limit, keeping input order."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in rows:
        rid = str(r["id"])
        if rid in seen:
            log(f"warning: duplicate id {rid!r} in prompts; keeping the first")
            continue
        if not row_matches(r, filters):
            continue
        if only_ids is not None and rid not in only_ids:
            continue
        if not in_shard(rid, shard):
            continue
        seen.add(rid)
        out.append(r)
        if limit is not None and len(out) >= limit:
            break
    return out


def effective_seed(row: dict[str, Any], attempt: int) -> int:
    """Row seed for attempt 0; for attempt k > 0 a seed derived from (row seed, k)."""
    base = (row.get("generation") or {}).get("seed")
    if base is None:
        base = stable_hash("seed", str(row["id"]), bits=31)
    base = int(base)
    return base if attempt == 0 else stable_hash("attempt", base, attempt, bits=31)


def row_watermark(row: dict[str, Any], no_watermark: bool) -> bool:
    return (not no_watermark) and row.get("watermark", True) is not False


# --------------------------------------------------------------------------- resumability


def load_done(out_path: Path) -> tuple[set[tuple[str, int, bool]], set[str]]:
    """(id, attempt, watermarked) triples already in ``out_path`` and the config_shas seen there.

    A truncated last line (crash mid-write) is cut off so appending continues on a clean line.
    """
    done: set[tuple[str, int, bool]] = set()
    shas: set[str] = set()
    if not out_path.exists():
        return done, shas
    data = out_path.read_bytes()
    lines = data.split(b"\n")
    offset = 0
    keep = len(data)
    for idx, line in enumerate(lines):
        if line.strip():
            try:
                r = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if idx == len(lines) - 1:  # last line without newline: a truncated write
                    keep = offset
                    break
                raise ValueError(f"{out_path}: corrupt line {idx + 1}") from None
            done.add((str(r["id"]), int(r.get("attempt", 0)), bool(r.get("watermarked", True))))
            if r.get("config_sha"):
                shas.add(r["config_sha"])
        offset += len(line) + 1
    if keep < len(data):
        log(f"warning: dropping a truncated last line of {out_path} ({len(data) - keep} bytes)")
        with open(out_path, "r+b") as f:
            f.truncate(keep)
    elif data and not data.endswith(b"\n"):
        with open(out_path, "ab") as f:
            f.write(b"\n")
    return done, shas
    data = out_path.read_bytes()
    good_end = 0
    pos = 0
    for line in data.split(b"\n"):
        end = pos + len(line) + 1
        if line.strip():
            try:
                r = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                if end >= len(data):  # last line, no newline: truncated write
                    break
                raise ValueError(f"{out_path}: corrupt line at byte {pos}") from None
            done.add((str(r["id"]), int(r.get("attempt", 0)), bool(r.get("watermarked", True))))
            if r.get("config_sha"):
                shas.add(r["config_sha"])
        good_end = min(end, len(data))
        pos = end
    if good_end < len(data):
        log(f"warning: dropping a truncated last line of {out_path} ({len(data) - good_end} bytes)")
        with open(out_path, "r+b") as f:
            f.truncate(good_end)
    elif data and not data.endswith(b"\n"):
        with open(out_path, "ab") as f:
            f.write(b"\n")
    return done, shas


# --------------------------------------------------------------------------- requests and batching


@dataclass
class Request:
    row: dict[str, Any]
    id: str
    prompt_text: str
    prompt_len: int
    max_new_tokens: int
    min_new_tokens: int | None
    watermark: bool
    seed: int


def chat_prompt(tok, row: dict[str, Any], enable_thinking_default: bool) -> str:
    """apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **chat_template_kwargs).

    ``enable_thinking`` defaults to the config value (False for Qwen3); a row's own
    ``chat_template_kwargs`` are merged on top.
    """
    if not getattr(tok, "chat_template", None):
        raise ValueError("tokenizer has no chat_template; prompt rows carry chat messages")
    kwargs = {"enable_thinking": enable_thinking_default}
    kwargs.update(row.get("chat_template_kwargs") or {})
    return tok.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False, **kwargs)


def build_requests(rows: Sequence[dict[str, Any]], tok, cfg: Config, attempt: int,
                   no_watermark: bool) -> list[Request]:
    reqs: list[Request] = []
    warned = False
    for r in rows:
        text = chat_prompt(tok, r, cfg.generation.enable_thinking)
        if (r.get("chat_template_kwargs") or {}).get("enable_thinking") and not warned:
            log("warning: some rows set enable_thinking=true; Qwen3 will emit <think> blocks (stripped from text)")
            warned = True
        n = len(tok(text, add_special_tokens=False)["input_ids"])
        mnt = r.get("max_new_tokens") or cfg.generation.max_new_tokens
        mn = r.get("min_new_tokens")
        reqs.append(Request(row=r, id=str(r["id"]), prompt_text=text, prompt_len=n, max_new_tokens=int(mnt),
                            min_new_tokens=int(mn) if mn else None, watermark=row_watermark(r, no_watermark),
                            seed=effective_seed(r, attempt)))
    return reqs


def plan_batches(reqs: Sequence[Request], batch_size: int, token_budget: int) -> list[list[Request]]:
    """Group by (max_new_tokens, min_new_tokens, watermark), largest max_new_tokens first; inside a group
    sort by prompt length (longest first) and cut greedily so that
    ``len(batch) * (longest prompt in batch + max_new_tokens) <= token_budget`` and ``len(batch) <= batch_size``
    (a single row always forms a batch, even over budget)."""
    groups: dict[tuple[int, int, bool], list[Request]] = {}
    for r in reqs:
        groups.setdefault((r.max_new_tokens, r.min_new_tokens or 0, r.watermark), []).append(r)
    batches: list[list[Request]] = []
    for key in sorted(groups, key=lambda k: (-k[0], -k[1], not k[2])):
        g = sorted(groups[key], key=lambda r: (-r.prompt_len, r.id))
        i = 0
        while i < len(g):
            per_row = g[i].prompt_len + key[0]
            cap = max(1, min(batch_size, token_budget // per_row))
            batches.append(g[i:i + cap])
            i += cap
    return batches


def batch_seed(batch: Sequence[Request], attempt: int) -> int:
    return stable_hash("batch", sorted(r.seed for r in batch), attempt, bits=63)


# --------------------------------------------------------------------------- generation


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)


def strip_think(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks. An unclosed <think> drops the rest; a lone </think> drops what precedes it."""
    if "<think>" not in text and "</think>" not in text:
        return text, False
    t = _THINK_BLOCK.sub("", text)
    if "</think>" in t:
        t = t.split("</think>", 1)[1].lstrip()
    if "<think>" in t:
        t = t[: t.index("<think>")]
    return t, True


def cut_at_eos(ids: Sequence[int], stop: set[int]) -> tuple[list[int], str]:
    out: list[int] = []
    for t in ids:
        out.append(t)
        if t in stop:
            return out, "eos"
    return out, "length"


def is_oom(e: BaseException) -> bool:
    oom_cls = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_cls is not None and isinstance(e, oom_cls)) or "out of memory" in str(e).lower()


def run_with_oom_retry(batch: list[Request], fn: Callable[[list[Request]], Any], cap: list[int]) -> None:
    """Call ``fn(batch)``; on CUDA OOM halve the batch, retry both halves, and lower ``cap[0]`` (the
    largest batch size to try from now on) so later batches do not hit the same OOM first.

    The retry happens outside the ``except`` block so the traceback (which pins GPU tensors) is freed first.
    Each half is re-seeded from its own rows (see batch_seed), so an OOM split changes the samples.
    """
    try:
        fn(batch)
        return
    except RuntimeError as e:
        if not is_oom(e) or len(batch) == 1:
            raise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    half = len(batch) // 2
    cap[0] = min(cap[0], max(1, half))
    log(f"CUDA OOM at batch size {len(batch)}; retrying as {half} + {len(batch) - half}")
    run_with_oom_retry(batch[:half], fn, cap)
    run_with_oom_retry(batch[half:], fn, cap)


class Runner:
    """Holds model/tokenizer/config and turns a batch of Requests into output rows."""

    def __init__(self, model, tok, device, cfg: Config, attempt: int):
        self.model, self.tok, self.device, self.cfg, self.attempt = model, tok, device, cfg, attempt
        self.stop = eos_ids(tok, model)
        self.sha = config_sha(cfg)
        self.wm_config = PortableSynthIDConfig.from_watermark_config(cfg.watermark)

    @torch.no_grad()
    def generate(self, batch: list[Request]) -> list[dict[str, Any]]:
        gen = self.cfg.generation
        first = batch[0]
        assert all((r.max_new_tokens, r.min_new_tokens, r.watermark) ==
                   (first.max_new_tokens, first.min_new_tokens, first.watermark) for r in batch)
        seed = batch_seed(batch, self.attempt)
        torch.manual_seed(seed)
        enc = self.tok([r.prompt_text for r in batch], return_tensors="pt", padding=True,
                       add_special_tokens=False).to(self.device)
        kwargs: dict[str, Any] = dict(max_new_tokens=first.max_new_tokens, do_sample=gen.do_sample,
                                      pad_token_id=self.tok.pad_token_id, eos_token_id=sorted(self.stop))
        if first.min_new_tokens:
            kwargs["min_new_tokens"] = first.min_new_tokens
        if gen.do_sample:
            kwargs.update(temperature=gen.temperature, top_k=gen.top_k, top_p=gen.top_p)
        if first.watermark:
            kwargs["watermarking_config"] = self.wm_config
        t0 = time.perf_counter()
        out = self.model.generate(**enc, **kwargs)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        new = out[:, enc["input_ids"].shape[1]:].cpu().tolist()
        now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
        rows = []
        for req, seq in zip(batch, new):
            ids, finish = cut_at_eos(seq, self.stop)
            text, had_think = strip_think(self.tok.decode(ids, skip_special_tokens=True))
            rec = dict(req.row)
            rec.update(
                text=text, token_ids=ids, n_new_tokens=len(ids), finish_reason=finish, had_think=had_think,
                attempt=self.attempt, watermarked=req.watermark, seed_effective=req.seed, batch_seed=seed,
                batch_size=len(batch), prompt_tokens=req.prompt_len,
                max_new_tokens_effective=req.max_new_tokens, min_new_tokens_effective=req.min_new_tokens,
                model_id=self.cfg.model.model_id, config_sha=self.sha, seconds=round(dt, 3), gen_time_utc=now,
            )
            rows.append(rec)
        return rows


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", nargs="+", required=True, help="prompt JSONL file(s)")
    ap.add_argument("--out", required=True, help="output JSONL (appended to; resumable)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="run YAML (default: promptset/config/generation.yaml)")
    ap.add_argument("--model", help="override model.model_id (HF id or local directory)")
    ap.add_argument("--device", help="override model.device (auto | cuda | cuda:1 | cpu)")
    ap.add_argument("--dtype", help="override model.dtype (bfloat16 | float16 | float32 | auto)")
    ap.add_argument("--filter", action="append", default=[], metavar="KEY=VALUE",
                    help="keep rows with row[KEY]==VALUE; repeat a key for OR, different keys are ANDed")
    ap.add_argument("--only-ids", help="file with ids to keep (one per line, or JSONL with id)")
    ap.add_argument("--shard", help="i/n: keep the i-th of n hash partitions of the ids (0-based)")
    ap.add_argument("--limit", type=int, help="keep the first N selected rows (before skipping done rows)")
    ap.add_argument("--attempt", type=int, default=0, help="attempt number; >0 derives new seeds (default 0)")
    ap.add_argument("--no-watermark", action="store_true", help="control run without the watermark")
    ap.add_argument("--batch-size", type=int, help="override batching.batch_size")
    ap.add_argument("--token-budget", type=int, help="override batching.token_budget")
    ap.add_argument("--save-config", help="also write the effective config YAML here")
    ap.add_argument("--allow-config-change", action="store_true",
                    help="append to an --out produced with a different watermark/sampling config (not recommended)")
    ap.add_argument("--dry-run", action="store_true", help="load the tokenizer only, print the batch plan, exit")
    args = ap.parse_args(argv)
    if args.attempt < 0:
        ap.error("--attempt must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg, batching = load_run_config(args.config, {
        "model.model_id": args.model, "model.device": args.device, "model.dtype": args.dtype,
        "batching.batch_size": args.batch_size, "batching.token_budget": args.token_budget,
    })
    sha = config_sha(cfg)

    rows = read_jsonl_files(args.prompts)
    only = read_id_list(args.only_ids) if args.only_ids else None
    selected = select_rows(rows, parse_filters(args.filter), only, parse_shard(args.shard), args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, shas = load_done(out_path)
    if shas - {sha} and not args.allow_config_change:
        raise SystemExit(f"{out_path} has rows from config_sha {sorted(shas - {sha})}, this run is {sha}. "
                         "Use another --out (or --allow-config-change).")
    todo = [r for r in selected
            if (str(r["id"]), args.attempt, row_watermark(r, args.no_watermark)) not in done]
    log(f"{len(rows)} prompt rows, {len(selected)} selected, {len(selected) - len(todo)} already in {out_path}, "
        f"{len(todo)} to generate (attempt {args.attempt}, config_sha {sha})")

    cfg_text = dump_config(cfg)
    cfg_paths = [sibling(out_path, ".config.yaml")] + ([Path(args.save_config)] if args.save_config else [])
    if not args.dry_run:
        for p in cfg_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists() and not args.allow_config_change:
                old_cfg, _ = load_run_config(p)
                if config_sha(old_cfg) != sha or old_cfg.model.model_id != cfg.model.model_id:
                    raise SystemExit(f"{p} holds a different watermark/sampling config or model; "
                                     "use another --out or --allow-config-change")
            p.write_text(cfg_text, encoding="utf-8")
    if not todo:
        log("nothing to do")
        return 0

    if args.dry_run:
        tok, model, device = load_tokenizer(cfg.model.model_id), None, torch.device("cpu")
    else:
        log(f"loading {cfg.model.model_id} (device={cfg.model.device}, dtype={cfg.model.dtype}) ...")
        model, tok, device = load_model(cfg.model)
        tok.padding_side = "left"
    reqs = build_requests(todo, tok, cfg, args.attempt, args.no_watermark)
    batches = plan_batches(reqs, batching.batch_size, batching.token_budget)
    planned = sum(len(b) * (b[0].prompt_len + b[0].max_new_tokens) for b in batches)
    log(f"{len(batches)} batches (batch_size<={batching.batch_size}, token_budget={batching.token_budget}); "
        f"padded token slots {planned:,}; watermark rows {sum(r.watermark for r in reqs)}/{len(reqs)}")
    if args.dry_run:
        for i, b in enumerate(batches):
            log(f"  batch {i}: rows={len(b)} prompt_len<={b[0].prompt_len} max_new={b[0].max_new_tokens} "
                f"min_new={b[0].min_new_tokens} wm={b[0].watermark}")
        return 0

    log(f"device={device} dtype={model.dtype} keys={cfg.watermark.depth} ngram_len={cfg.watermark.ngram_len} "
        f"T={cfg.generation.temperature} top_k={cfg.generation.top_k} top_p={cfg.generation.top_p}")
    runner = Runner(model, tok, device, cfg, args.attempt)
    stats = {"rows": 0, "tokens": 0, "seconds": 0.0}
    t_start = time.perf_counter()
    cap = [batching.batch_size]

    with open(out_path, "a", encoding="utf-8") as fout:
        def do_batch(b: list[Request]) -> None:
            recs = runner.generate(b)
            for rec in recs:
                fout.write(dumps(rec) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
            ntok = sum(r["n_new_tokens"] for r in recs)
            dt = recs[0]["seconds"]
            stats["rows"] += len(recs)
            stats["tokens"] += ntok
            stats["seconds"] += dt
            n_eos = sum(r["finish_reason"] == "eos" for r in recs)
            elapsed = time.perf_counter() - t_start
            eta = elapsed / stats["rows"] * (len(reqs) - stats["rows"])
            log(f"[{stats['rows']}/{len(reqs)}] batch rows={len(recs)} prompt<={b[0].prompt_len} "
                f"max_new={b[0].max_new_tokens} new_tok={ntok} eos={n_eos}/{len(recs)} {dt:.1f}s "
                f"{ntok / max(dt, 1e-9):.0f} tok/s | total {stats['tokens'] / max(elapsed, 1e-9):.0f} tok/s, "
                f"ETA {eta / 60:.1f} min")

        for b in batches:
            for i in range(0, len(b), cap[0]):
                run_with_oom_retry(b[i:i + cap[0]], do_batch, cap)

    elapsed = time.perf_counter() - t_start
    log(f"done: {stats['rows']} rows, {stats['tokens']:,} new tokens in {elapsed:.1f}s "
        f"({stats['tokens'] / max(elapsed, 1e-9):.0f} tok/s)")
    manifest = {
        "time_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "argv": list(argv) if argv is not None else sys.argv[1:], "config_sha": sha,
        "rows": stats["rows"], "new_tokens": stats["tokens"], "seconds": round(elapsed, 1),
        "attempt": args.attempt, "host": platform.node(), "python": platform.python_version(),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "batching": dataclasses.asdict(batching),
    }
    with open(sibling(out_path, ".runs.jsonl"), "a", encoding="utf-8") as f:
        f.write(dumps(manifest) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
