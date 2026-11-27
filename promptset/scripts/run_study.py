"""The whole GPU job in one command: generate -> trim -> regenerate -> score (AI and human) -> report.

    python promptset/scripts/run_study.py --study promptset/config/study.yaml
    python promptset/scripts/run_study.py --study promptset/config/study.yaml --set filters='[split=pilot,main]'
    python promptset/scripts/run_study.py --dump dump/study --plan        # what would run, nothing is started

It drives the existing scripts (generate.py, trim.py, build_edit_prompts.py build-outline,
verify_generations.py, load_dump.py report) as subprocesses, one generation process per GPU, and writes
everything into one dump directory (layout in promptset/JOB.md). It is

* **resumable**: a killed job (crash, preemption, SIGTERM) is continued by running the same command again.
  generate.py skips rows already written (per id, attempt and watermark flag, across all shard files, so the
  GPU count may change between runs); a half-written last line is dropped.
* **idempotent**: trimming, outline building and scoring are skipped when their inputs and settings have not
  changed since they last ran (fingerprints in dump/state/). A finished dump re-run does nothing new.
* **safe against mixing**: the effective run config (watermark keys + sampling) is frozen in
  dump/config/run_config.yaml on the first run; a later run with other keys or sampling is refused.

Stages, in order: preflight; for each stream (ox, wm, ctrl by default): generate attempt 0, then
[trim, regenerate length failures as attempt k] up to regen.max_attempts, final trim (the ox stream is not
trimmed; its outlines feed the "outline" stage that builds the O_outline prompts for wm); human (trim each
human set); score (every trimmed file with per-token details, plus every raw generation); report.

Failures never stop the job unless nothing can run: a generation process that crashes is restarted
(generate.shard_retries) and then recorded; a batch that fails inside generate.py is logged with its ids and
traceback (dump/failures/gen.<stream>.jsonl) and retried on the next run; ids that still fail the length
rule after the last attempt go to dump/failures/dropped.<stream>.jsonl; prompt rows that never produced a
generation go to dump/failures/missing.<stream>.txt. The exit code is 0 only if every stage succeeded.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import yaml

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
from _common import dumps, iter_jsonl, log, parse_filters, read_id_list, row_matches, stable_hash  # noqa: E402

DEFAULT_STUDY = REPO / "promptset" / "config" / "study.yaml"
GEN_FILE = re.compile(r"^a(\d+)\.s(\d+)of(\d+)\.jsonl$")
MIN_ROWS_PER_SHARD = 32


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- config


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def load_study(path: str | Path, sets: Sequence[str]) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for spec in sets:
        k, sep, v = spec.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {spec!r}")
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    if os.environ.get("RUNCFG"):
        cfg["run_config"] = os.environ["RUNCFG"]
    cfg.setdefault("streams", {})
    cfg.setdefault("prompts", {})
    cfg.setdefault("human", {})
    for sec in ("regen", "trim", "detection", "generate"):
        cfg[sec] = cfg.get(sec) or {}
    return cfg


def rpath(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(p))))
    return p if p.is_absolute() else REPO / p


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_if_changed(path: Path, text: str) -> None:
    """Write only when the content differs, so re-runs keep mtimes (and fingerprints) stable."""
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def fingerprint(paths: Sequence[Path], params: Any) -> str:
    """Cheap identity of a stage's inputs: (path, size, mtime) of every file + the stage parameters."""
    items = []
    for p in sorted(set(paths)):
        st = p.stat() if p.exists() else None
        items.append([str(p), st.st_size if st else None, st.st_mtime_ns if st else None])
    return f"{stable_hash(items, params, bits=64):016x}"


# --------------------------------------------------------------------------- environment


def visible_gpus(spec: Any) -> list[str] | None:
    """GPU ids to use, one process each; None = a single process on model.device (CPU, MPS or 1 GPU)."""
    if spec in (0, "0", None, False):
        return None
    ids: list[str] = []
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        ids = [x.strip() for x in env.split(",") if x.strip() and x.strip() != "-1"]
    else:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=30)
            ids = [x.strip() for x in out.stdout.splitlines() if x.strip()] if out.returncode == 0 else []
        except (OSError, subprocess.TimeoutExpired):
            ids = []
    if spec != "auto":
        ids = ids[: int(spec)]
    return ids or None


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {"host": platform.node(), "python": platform.python_version(),
                            "platform": platform.platform(), "cpu_count": os.cpu_count()}
    for mod in ("torch", "transformers", "tokenizers", "accelerate"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        info["gpus"] = out.stdout.strip().splitlines() if out.returncode == 0 else []
    except (OSError, subprocess.TimeoutExpired):
        info["gpus"] = []
    info["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    for k in ("SLURM_JOB_ID", "SLURM_JOB_NAME", "PBS_JOBID", "HF_HOME"):
        if os.environ.get(k):
            info[k] = os.environ[k]
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True)
        info["git_commit"] = head.stdout.strip() or None
        info["git_dirty_files"] = [l[3:] for l in dirty.stdout.splitlines()]
    except OSError:
        info["git_commit"] = None
    return info


# --------------------------------------------------------------------------- the job


class Job:
    def __init__(self, study: dict[str, Any], study_path: Path, dump: Path, args: argparse.Namespace):
        self.s = study
        self.study_path = study_path
        self.dump = dump
        self.args = args
        self.py = study.get("python") or sys.executable
        self.stages: list[dict[str, Any]] = []
        self.failed = False
        self.children: list[subprocess.Popen] = []
        for d in ("config", "gen", "trim", "scores", "tokens", "failures", "logs", "state", "prompts", "results"):
            (dump / d).mkdir(parents=True, exist_ok=True)
        self.run_config = dump / "config" / "run_config.yaml"
        self.tokenizer: str = ""
        self.batching: dict[str, Any] = {}
        self.gpus: list[str] | None = None
        self._rows_cache: dict[Path, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------ helpers

    def record(self, stage: str, status: str, t0: float, **detail: Any) -> dict[str, Any]:
        rec = {"stage": stage, "status": status, "seconds": round(time.perf_counter() - t0, 1), **detail}
        self.stages.append(rec)
        if status == "failed":
            self.failed = True
        log(f"[{now()}] {stage}: {status} " + json.dumps(detail, ensure_ascii=False)[:400])
        return rec

    def fail(self, stage: str, **detail: Any) -> None:
        with open(self.dump / "failures" / "job.jsonl", "a", encoding="utf-8") as f:
            f.write(dumps({"time_utc": now(), "stage": stage, **detail}) + "\n")

    def run(self, cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {now()} $ {shlex.join(cmd)}\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO),
                                 env={**os.environ, **(env or {})})
            self.children.append(p)
            try:
                return p.wait()
            finally:
                self.children.remove(p)

    def run_parallel(self, jobs: list[tuple[list[str], Path, dict[str, str]]]) -> list[int]:
        procs = []
        for cmd, log_path, env in jobs:
            lf = open(log_path, "a", encoding="utf-8")
            lf.write(f"\n===== {now()} $ {shlex.join(cmd)}\n")
            lf.flush()
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO),
                                 env={**os.environ, **env})
            self.children.append(p)
            procs.append((p, lf))
        codes = []
        for p, lf in procs:
            codes.append(p.wait())
            lf.close()
            self.children.remove(p)
        return codes

    def terminate_children(self) -> None:
        for p in list(self.children):
            with contextlib.suppress(Exception):
                p.terminate()
        for p in list(self.children):
            with contextlib.suppress(Exception):
                p.wait(timeout=60)

    def state_ok(self, name: str, fp: str, outputs: Sequence[Path]) -> bool:
        st = self.dump / "state" / f"{name}.json"
        if self.args.force or not st.exists() or not all(o.exists() for o in outputs):
            return False
        try:
            return json.loads(st.read_text(encoding="utf-8")).get("fingerprint") == fp
        except json.JSONDecodeError:
            return False

    def state_save(self, name: str, fp: str, **extra: Any) -> None:
        (self.dump / "state" / f"{name}.json").write_text(
            json.dumps({"fingerprint": fp, "time_utc": now(), **extra}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    def rows_of(self, path: Path) -> list[dict[str, Any]]:
        if path not in self._rows_cache:
            self._rows_cache[path] = list(iter_jsonl(path))
        return self._rows_cache[path]

    # ------------------------------------------------------------------ preflight

    def preflight(self) -> None:
        t0 = time.perf_counter()
        import generate as G

        src = rpath(self.s.get("run_config"))
        if not src or not src.exists():
            raise SystemExit(f"run_config not found: {src}")
        overrides = {"model.model_id": self.s.get("model")}
        cfg, batching = G.load_run_config(src, overrides)
        g = self.s["generate"]
        self.batching = {"batch_size": g.get("batch_size") or batching.batch_size,
                         "token_budget": g.get("token_budget") or batching.token_budget}
        sha = G.config_sha(cfg)
        if cfg.watermark.keys == list(__import__("synthid_wm.config", fromlist=["BLOG_KEYS"]).BLOG_KEYS):
            log("WARNING: run_config uses the PUBLIC demo watermark keys (GPU_RUNBOOK.md step 1 makes private ones)")
        if self.run_config.exists():
            old, _ = G.load_run_config(self.run_config)
            if G.config_sha(old) != sha or old.model.model_id != cfg.model.model_id:
                raise SystemExit(
                    f"{self.run_config} (frozen on this dump's first run) has config_sha {G.config_sha(old)} / "
                    f"model {old.model.model_id}; this run has {sha} / {cfg.model.model_id}. Use another dump_dir.")
        # Detection settings may differ between runs (they only change scoring); keep the frozen file's
        # watermark/sampling and refresh the rest.
        text = G.dump_config(cfg)
        if not self.run_config.exists() or self.run_config.read_text(encoding="utf-8") != text:
            self.run_config.write_text(text, encoding="utf-8")   # unchanged -> keep mtime (score fingerprints)
        self.tokenizer = str(self.s.get("tokenizer") or cfg.model.model_id)
        self.gpus = visible_gpus(self.s.get("gpus", "auto"))
        inputs = {}
        for name, p in list((self.s.get("prompts") or {}).items()) + [
                (f"human.{k}", v) for k, v in (self.s.get("human") or {}).items()]:
            rp = rpath(p)
            inputs[name] = {"path": str(rp), "exists": bool(rp and rp.exists()),
                            "sha256": sha256_file(rp) if rp and rp.exists() else None,
                            "bytes": rp.stat().st_size if rp and rp.exists() else None}
        write_if_changed(self.dump / "config" / "study.yaml", yaml.safe_dump(self.s, sort_keys=False, allow_unicode=True))
        self.manifest_base = {
            "dump": str(self.dump), "study_file": str(self.study_path), "run_config_source": str(src),
            "config_sha": sha, "model_id": cfg.model.model_id, "tokenizer": self.tokenizer,
            "gpus_used": self.gpus, "batching": self.batching, "inputs": inputs,
            "z_threshold": self.s["detection"].get("z_threshold", 3.0), "environment": environment(),
        }
        self.record("preflight", "done", t0, config_sha=sha, gpus=self.gpus, tokenizer=self.tokenizer,
                    missing_inputs=[k for k, v in inputs.items() if not v["exists"]])

    # ------------------------------------------------------------------ generation

    def stream_prompt_files(self, stream: str) -> list[Path]:
        files = []
        for src in self.s["streams"][stream].get("sources") or []:
            p = self.dump / "prompts" / "O_prompts.jsonl" if src == "O" else rpath(self.s["prompts"].get(src))
            if p is None:
                log(f"warning: stream {stream}: unknown source {src!r}")
            elif p.exists():
                files.append(p)
            elif src != "O":
                log(f"warning: stream {stream}: prompt file {p} ({src}) not found; skipped")
        return files

    def stream_filters(self, stream: str) -> list[str]:
        return list(self.s.get("filters") or []) + list(self.s["streams"][stream].get("filter") or [])

    def selected(self, stream: str) -> list[dict[str, Any]]:
        import generate as G

        rows: list[dict[str, Any]] = []
        for p in self.stream_prompt_files(stream):
            rows.extend(self.rows_of(p))
        return G.select_rows(rows, parse_filters(self.stream_filters(stream)), None, None, None)

    def gen_files(self, stream: str) -> list[Path]:
        d = self.dump / "gen" / stream
        return sorted(p for p in d.glob("a*.jsonl") if GEN_FILE.match(p.name)) if d.is_dir() else []

    def generate(self, stream: str, attempt: int, only_ids: set[str] | None) -> dict[str, Any]:
        import generate as G

        t0 = time.perf_counter()
        name = f"gen:{stream}:a{attempt}"
        sc = self.s["streams"][stream]
        no_wm = bool(sc.get("no_watermark"))
        sel = self.selected(stream)
        if only_ids is not None:
            sel = [r for r in sel if str(r["id"]) in only_ids]
        files = self.gen_files(stream)
        done = G.read_done_keys(files)
        todo = [r for r in sel if (str(r["id"]), attempt, G.row_watermark(r, no_wm)) not in done]
        if not todo:
            return self.record(name, "skipped", t0, selected=len(sel), todo=0)
        if self.args.plan:
            return self.record(name, "planned", t0, selected=len(sel), todo=len(todo))
        ids_file = None
        if only_ids is not None:
            ids_file = self.dump / "state" / f"ids.{stream}.a{attempt}.txt"
            write_if_changed(ids_file, "".join(f"{i}\n" for i in sorted(only_ids)))
        gpus = self.gpus
        n = 1 if gpus is None else max(1, min(len(gpus), math.ceil(len(todo) / MIN_ROWS_PER_SHARD)))
        g = self.s["generate"]
        base = [self.py, str(SCRIPTS / "generate.py"), "--config", str(self.run_config),
                "--batch-size", str(self.batching["batch_size"]), "--token-budget", str(self.batching["token_budget"]),
                "--attempt", str(attempt), "--continue-on-error",
                "--failures", str(self.dump / "failures" / f"gen.{stream}.jsonl"), "--prompts",
                *map(str, self.stream_prompt_files(stream))]
        for f in self.stream_filters(stream):
            base += ["--filter", f]
        if no_wm:
            base.append("--no-watermark")
        if ids_file:
            base += ["--only-ids", str(ids_file)]
        base += [str(x) for x in g.get("extra_args") or []]
        out_dir = self.dump / "gen" / stream
        out_dir.mkdir(parents=True, exist_ok=True)

        def cmd_for(i: int) -> tuple[list[str], Path, dict[str, str]]:
            out = out_dir / f"a{attempt}.s{i}of{n}.jsonl"
            others = [str(p) for p in self.gen_files(stream) if p != out]
            cmd = base + ["--out", str(out), "--shard", f"{i}/{n}", "--done-from", *others]
            env = {"CUDA_VISIBLE_DEVICES": gpus[i]} if gpus is not None else {}
            return cmd, self.dump / "logs" / f"gen.{stream}.a{attempt}.s{i}of{n}.log", env

        pending = list(range(n))
        codes: dict[int, int] = {}
        for round_ in range(int(g.get("shard_retries", 2)) + 1):
            if round_:
                log(f"{name}: restarting shard(s) {pending} (attempt {round_} of {g.get('shard_retries', 2)})")
            res = self.run_parallel([cmd_for(i) for i in pending])
            for i, c in zip(pending, res):
                codes[i] = c
            pending = [i for i in pending if codes[i] != 0]
            if not pending:
                break
        after = G.read_done_keys(self.gen_files(stream))
        still = [str(r["id"]) for r in todo if (str(r["id"]), attempt, G.row_watermark(r, no_wm)) not in after]
        status = "done" if not still and not pending else ("partial" if len(still) < len(todo) else "failed")
        if pending:
            self.fail(name, shards=pending, returncodes={i: codes[i] for i in pending},
                      logs=[str(cmd_for(i)[1]) for i in pending], ungenerated=len(still))
        return self.record(name, status, t0, selected=len(sel), todo=len(todo), generated=len(todo) - len(still),
                           shards=n, returncodes=codes)

    # ------------------------------------------------------------------ trimming

    def trim_stream(self, stream: str) -> dict[str, Any] | None:
        t0 = time.perf_counter()
        files = self.gen_files(stream)
        out = self.dump / "trim" / f"{stream}.jsonl"
        if not files:
            self.record(f"trim:{stream}", "skipped", t0, reason="no generations")
            return None
        t = self.s["trim"]
        cmd = [self.py, str(SCRIPTS / "trim.py"), "gen", "--in", *map(str, files), "--out", str(out),
               "--tokenizer", self.tokenizer,
               "--regen-reasons", ",".join(self.s["regen"].get("reasons") or ["too_short", "no_sentence_end", "empty"])]
        if t.get("min_frac") is not None:
            cmd += ["--min-frac", str(t["min_frac"])]
        if t.get("slack_words") is not None:
            cmd += ["--slack-words", str(t["slack_words"])]
        cmd += [str(x) for x in t.get("extra_args") or []]
        fp = fingerprint(files, cmd[3:])
        if self.state_ok(f"trim.{stream}", fp, [out]):
            return self.record(f"trim:{stream}", "skipped", t0, reason="unchanged")
        if self.args.plan:
            return self.record(f"trim:{stream}", "planned", t0, inputs=len(files))
        code = self.run(cmd, self.dump / "logs" / f"trim.{stream}.log")
        if code != 0:
            self.fail(f"trim:{stream}", returncode=code, log=str(self.dump / "logs" / f"trim.{stream}.log"))
            return self.record(f"trim:{stream}", "failed", t0, returncode=code)
        self.state_save(f"trim.{stream}", fp)
        meta = json.loads((self.dump / "trim" / f"{stream}.summary.json").read_text(encoding="utf-8"))["meta"]
        return self.record(f"trim:{stream}", "done", t0, passages=meta["n_passages"], accepted=meta["n_accepted"],
                           regen_ids=meta["n_regen_ids"])

    def regen_ids(self, stream: str, attempt: int) -> set[str]:
        p = self.dump / "trim" / f"{stream}.regen_ids.attempt{attempt}.txt"
        return read_id_list(p) if p.exists() else set()

    def write_dropped(self, stream: str) -> int:
        p = self.dump / "trim" / f"{stream}.regen_ids.txt"
        ids = read_id_list(p) if p.exists() else set()
        out = self.dump / "failures" / f"dropped.{stream}.jsonl"
        recs = []
        if ids:
            # the largest cutoff's rejection per id (what made it a regeneration candidate)
            best: dict[str, dict[str, Any]] = {}
            for r in iter_jsonl(self.dump / "trim" / f"{stream}.jsonl"):
                if str(r["id"]) in ids and not r.get("accepted"):
                    k = str(r["id"])
                    if k not in best or (r["cutoff_kind"], r["cutoff"]) > (best[k]["cutoff_kind"], best[k]["cutoff"]):
                        best[k] = r
            for i in sorted(ids):
                b = best.get(i, {})
                recs.append({"stream": stream, "id": i, "language": b.get("language"), "level": b.get("level"),
                             "reason": b.get("reject_reason"), "largest_cutoff": [b.get("cutoff_kind"), b.get("cutoff")],
                             "n_words": b.get("n_words"), "n_tokens": b.get("n_tokens"),
                             "attempts_tried": b.get("attempts_tried"),
                             "note": "largest cutoff still rejected after the last attempt; smaller cutoffs may be accepted"})
        write_if_changed(out, "".join(dumps(r) + "\n" for r in recs))
        return len(recs)

    def write_missing(self, stream: str) -> int:
        import generate as G

        no_wm = bool(self.s["streams"][stream].get("no_watermark"))
        have = {(k[0], k[2]) for k in G.read_done_keys(self.gen_files(stream))}
        sel = self.selected(stream)
        missing = sorted(str(r["id"]) for r in sel if (str(r["id"]), G.row_watermark(r, no_wm)) not in have)
        write_if_changed(self.dump / "prompts" / f"{stream}.selected_ids.txt", "".join(f"{r['id']}\n" for r in sel))
        write_if_changed(self.dump / "failures" / f"missing.{stream}.txt", "".join(f"{i}\n" for i in missing))
        return len(missing)

    def run_stream(self, stream: str) -> None:
        sc = self.s["streams"][stream]
        if not self.stream_prompt_files(stream):
            self.record(f"stream:{stream}", "skipped", time.perf_counter(), reason="no prompt files")
            return
        self.generate(stream, 0, None)
        if sc.get("trim", True) is False:
            if not self.args.plan:
                self.record(f"missing:{stream}", "done", time.perf_counter(), n=self.write_missing(stream))
            return
        max_k = int(self.s["regen"].get("max_attempts", 3))
        for k in range(1, max_k + 1):
            if self.args.plan:
                break
            self.trim_stream(stream)
            ids = self.regen_ids(stream, k)
            if not ids:
                break
            self.generate(stream, k, ids)
        if not self.args.plan:
            self.trim_stream(stream)
            n_drop = self.write_dropped(stream) if (self.dump / "trim" / f"{stream}.jsonl").exists() else 0
            self.record(f"missing:{stream}", "done", time.perf_counter(), missing=self.write_missing(stream),
                        dropped_after_attempts=n_drop)

    def outline(self) -> None:
        t0 = time.perf_counter()
        files = self.gen_files("ox")
        out = self.dump / "prompts" / "O_prompts.jsonl"
        if not files:
            self.record("outline", "skipped", t0, reason="no ox generations")
            return
        cmd = [self.py, str(SCRIPTS / "build_edit_prompts.py"), "build-outline", "--gen", *map(str, files),
               "--out", str(out)]
        fp = fingerprint(files, cmd[2:])
        if self.state_ok("outline", fp, [out]):
            self.record("outline", "skipped", t0, reason="unchanged")
            return
        if self.args.plan:
            self.record("outline", "planned", t0)
            return
        code = self.run(cmd, self.dump / "logs" / "outline.log")
        if code != 0:
            self.fail("outline", returncode=code, log=str(self.dump / "logs" / "outline.log"))
            self.record("outline", "failed", t0, returncode=code)
            return
        self._rows_cache.pop(out, None)
        self.state_save("outline", fp)
        self.record("outline", "done", t0, rows=sum(1 for _ in iter_jsonl(out)))

    # ------------------------------------------------------------------ human text

    def human(self) -> None:
        cut = self.s.get("human_cutoffs") or yaml.safe_load(
            (REPO / "promptset" / "config" / "promptset.yaml").read_text(encoding="utf-8"))["cutoffs"]
        for name, p in (self.s.get("human") or {}).items():
            t0 = time.perf_counter()
            src = rpath(p)
            if not src or not src.exists():
                self.record(f"human:{name}", "skipped", t0, reason=f"{src} not found")
                continue
            out = self.dump / "trim" / f"human.{name}.jsonl"
            t = self.s["trim"]
            cmd = [self.py, str(SCRIPTS / "trim.py"), "human", "--in", str(src), "--out", str(out),
                   "--tokenizer", self.tokenizer, "--words", *map(str, cut["words"]), "--tokens", *map(str, cut["tokens"])]
            if t.get("min_frac") is not None:
                cmd += ["--min-frac", str(t["min_frac"])]
            if t.get("slack_words") is not None:
                cmd += ["--slack-words", str(t["slack_words"])]
            fp = fingerprint([src], cmd[3:])
            if self.state_ok(f"human.{name}", fp, [out]):
                self.record(f"human:{name}", "skipped", t0, reason="unchanged")
                continue
            if self.args.plan:
                self.record(f"human:{name}", "planned", t0)
                continue
            code = self.run(cmd, self.dump / "logs" / f"human.{name}.log")
            if code != 0:
                self.fail(f"human:{name}", returncode=code, log=str(self.dump / "logs" / f"human.{name}.log"))
                self.record(f"human:{name}", "failed", t0, returncode=code)
                continue
            self.state_save(f"human.{name}", fp)
            meta = json.loads((self.dump / "trim" / f"human.{name}.summary.json").read_text(encoding="utf-8"))["meta"]
            self.record(f"human:{name}", "done", t0, passages=meta["n_passages"], accepted=meta["n_accepted"])

    # ------------------------------------------------------------------ scoring

    def workers(self) -> int:
        w = self.s["detection"].get("workers", "auto")
        return min(16, os.cpu_count() or 1) if w in (None, "auto") else int(w)

    def score(self, label: str, inputs: list[Path], out: Path, tags: dict[str, str], extra: list[str],
              details: Path | None) -> None:
        t0 = time.perf_counter()
        det = self.s["detection"]
        cmd = [self.py, str(SCRIPTS / "verify_generations.py"), "--config", str(self.run_config),
               "--tokenizer", self.tokenizer, "--in", *map(str, inputs), "--out", str(out),
               "--z-threshold", str(det.get("z_threshold", 3.0)), "--workers", str(self.workers()), *extra]
        if det.get("min_scored") is not None:
            cmd += ["--min-scored", str(det["min_scored"])]
        for k, v in tags.items():
            cmd += ["--tag", f"{k}={v}"]
        if details is not None:
            cmd += ["--details", str(details)]
        # the run config's detection block may be refreshed between runs: include its content
        fp = fingerprint(inputs + [self.run_config], cmd[2:])
        outs = [out] + ([details] if details else [])
        if self.state_ok(f"score.{label}", fp, outs):
            self.record(f"score:{label}", "skipped", t0, reason="unchanged")
            return
        if self.args.plan:
            self.record(f"score:{label}", "planned", t0)
            return
        code = self.run(cmd, self.dump / "logs" / f"score.{label}.log")
        if code != 0:
            self.fail(f"score:{label}", returncode=code, log=str(self.dump / "logs" / f"score.{label}.log"))
            self.record(f"score:{label}", "failed", t0, returncode=code)
            return
        self.state_save(f"score.{label}", fp)
        self.record(f"score:{label}", "done", t0, rows=sum(1 for _ in iter_jsonl(out)))

    def score_all(self) -> None:
        want_details = bool(self.s["detection"].get("details", True))
        for stream, sc in self.s["streams"].items():
            if sc.get("trim", True) is False:
                continue
            trimmed = self.dump / "trim" / f"{stream}.jsonl"
            if trimmed.exists():
                self.score(stream, [trimmed], self.dump / "scores" / f"{stream}.jsonl",
                           {"stream": stream, "view": "passage"}, [],
                           self.dump / "tokens" / f"{stream}.jsonl.gz" if want_details else None)
            files = self.gen_files(stream)
            if files and self.s["detection"].get("score_raw", True):
                self.score(f"raw.{stream}", files, self.dump / "scores" / f"raw.{stream}.jsonl",
                           {"stream": stream, "view": "raw"}, ["--field", "text", "--score-ids"], None)
        for name in (self.s.get("human") or {}):
            trimmed = self.dump / "trim" / f"human.{name}.jsonl"
            if trimmed.exists():
                self.score(f"human.{name}", [trimmed], self.dump / "scores" / f"human.{name}.jsonl",
                           {"stream": f"human.{name}", "view": "passage"}, [],
                           self.dump / "tokens" / f"human.{name}.jsonl.gz" if want_details else None)

    def report(self) -> None:
        t0 = time.perf_counter()
        if self.args.plan:
            self.record("report", "planned", t0)
            return
        cmd = [self.py, str(SCRIPTS / "load_dump.py"), "report", str(self.dump),
               "--z-threshold", str(self.s["detection"].get("z_threshold", 3.0))]
        code = self.run(cmd, self.dump / "logs" / "report.log")
        if code != 0:
            self.fail("report", returncode=code, log=str(self.dump / "logs" / "report.log"))
        self.record("report", "done" if code == 0 else "failed", t0, out=str(self.dump / "report.md"))

    # ------------------------------------------------------------------ main flow

    def write_manifest(self, status: str, started: str) -> None:
        man = {**getattr(self, "manifest_base", {}), "status": status, "started_utc": started,
               "finished_utc": now(), "argv": sys.argv, "stages": self.stages}
        (self.dump / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(self.dump / "job_runs.jsonl", "a", encoding="utf-8") as f:
            f.write(dumps({"started_utc": started, "finished_utc": man["finished_utc"], "status": status,
                           "argv": sys.argv, "host": platform.node(),
                           "stages": [{k: s[k] for k in ("stage", "status", "seconds")} for s in self.stages]}) + "\n")

    def main(self) -> int:
        started = now()
        status = "failed"
        try:
            self.preflight()
            streams = list(self.s["streams"])
            # ox first (its outlines are the O prompts of the other streams)
            for stream in sorted(streams, key=lambda x: (x != "ox", streams.index(x))):
                self.run_stream(stream)
                if stream == "ox":
                    self.outline()
            self.human()
            self.score_all()
            self.report()
            status = "planned" if self.args.plan else ("done_with_failures" if self.failed else "done")
        except KeyboardInterrupt:
            status = "interrupted"
            log("interrupted: stopping child processes; re-run the same command to resume")
            self.terminate_children()
        finally:
            if not self.args.plan:
                self.write_manifest(status, started)
            log(f"[{now()}] job {status}; dump {self.dump}")
        return {"done": 0, "planned": 0, "interrupted": 130}.get(status, 1)


@contextlib.contextmanager
def dump_lock(dump: Path) -> Iterator[None]:
    dump.mkdir(parents=True, exist_ok=True)
    f = open(dump / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another job is running on {dump} (lock {dump / '.lock'})") from None
    try:
        f.write(f"{os.getpid()} {platform.node()} {now()}\n")
        f.flush()
        yield
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default=str(DEFAULT_STUDY), help="job config (default promptset/config/study.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=YAML", help="override a study key")
    ap.add_argument("--dump", help="override dump_dir")
    ap.add_argument("--plan", action="store_true", help="show what would run; start nothing")
    ap.add_argument("--force", action="store_true", help="redo trimming / outline / scoring even if unchanged")
    args = ap.parse_args(argv)
    study_path = Path(args.study)
    study = load_study(study_path, args.set)
    dump = rpath(args.dump or study.get("dump_dir") or "dump/study")
    assert dump is not None
    with dump_lock(dump):
        job = Job(study, study_path, dump, args)

        def on_term(signum, frame):  # SIGTERM (scheduler preemption / scancel) -> clean stop, resumable
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, on_term)
        return job.main()


if __name__ == "__main__":
    raise SystemExit(main())
