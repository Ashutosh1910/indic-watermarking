"""Live status of a run_study.py dump: progress, speed, ETA, output health and a quick watermark check.

run_study.py prints this block into the job log (logs/ashutosh.<jobid>.out) every ``watch.every_min`` minutes
while it generates, and writes the latest one to ``<dump>/status.txt``, so a login node only needs ``cat``.
By hand (on the compute node of a running job, or anywhere with the venv):

    python promptset/scripts/watch_study.py dump/main            # status block, exit 0
    python promptset/scripts/watch_study.py dump/main --no-score # no tokenizer, progress and health only

What it reads: the generation shards (``<dump>/gen/<stream>/a<attempt>.s<i>of<n>.jsonl``), their logs
(``<dump>/logs/gen.*.log``: the last ``[done/total] ... tok/s, ETA`` line) and the failure files. It writes
nothing else and never prints passage text.

Checks (each one prints a ``WARN`` line; ``STATUS WARN`` at the end means look before the run goes on):

* a shard log has not moved for ``stall_min`` minutes while its shard is not done;
* failed batches or failed stages;
* fewer than ``min_eos`` of a language's rows stopped at EOS (runaway or repeating outputs);
* Hindi rows whose letters are mostly not Devanagari (the model switched language);
* <think> blocks in outputs;
* watermarked 100 %-AI rows (G, G_title, O_outline) with mean z below ``min_wm_z`` once ``min_rows`` are in,
  or unwatermarked rows (ctrl stream, OX) with mean z above ``max_null_z``: wrong keys or a broken processor.
  The z check scores the newest ``sample`` rows per stream and language from their sampled token ids.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import add_synthid_to_path  # noqa: E402

GEN_FILE = re.compile(r"^a(\d+)\.s(\d+)of(\d+)\.jsonl$")
PROGRESS = re.compile(r"^\[(\d+)/(\d+)\].*?total (\d+) tok/s, ETA ([\d.]+) min")
FULL_AI = {"G", "G_title", "O_outline"}
DEFAULTS = {"every_min": 15, "sample": 40, "stall_min": 30, "min_eos": 0.5, "min_hi_script": 0.8,
            "min_wm_z": 2.0, "max_null_z": 1.5, "min_rows": 10, "min_scored": 50}
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LETTER = re.compile(r"[^\W\d_]")


def devanagari_share(text: str) -> float:
    letters = LETTER.findall(text or "")
    return sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters) if letters else 0.0


class _Shard:
    def __init__(self) -> None:
        self.offset = 0
        self.rows = 0
        self.by_lang: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self.recent: dict[tuple[str, bool, bool], deque] = defaultdict(lambda: deque(maxlen=200))


class Watcher:
    """Reads the dump incrementally (byte offsets per shard), so calling it every few minutes stays cheap."""

    def __init__(self, dump: str | Path, settings: dict[str, Any] | None = None, score: bool = True) -> None:
        self.dump = Path(dump)
        self.s = {**DEFAULTS, **(settings or {})}
        self.score = score
        self.shards: dict[Path, _Shard] = {}
        self._verifiers: dict[Path, Any] = {}
        self._tok = None

    # ------------------------------------------------------------------ reading

    def _read(self, path: Path) -> _Shard:
        sh = self.shards.setdefault(path, _Shard())
        with open(path, "rb") as f:
            f.seek(sh.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1                    # a half-written last line waits for the next call
        for line in data[:end].splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            sh.rows += 1
            lang = str(r.get("language"))
            st = sh.by_lang[lang]
            st["rows"] += 1
            st["eos"] += r.get("finish_reason") == "eos"
            st["think"] += bool(r.get("had_think"))
            st["empty"] += not (r.get("text") or "").strip()
            st["new_tokens"] += int(r.get("n_new_tokens") or 0)
            if lang == "hi":
                st["off_script"] += devanagari_share(r.get("text") or "") < self.s["min_hi_script"]
            wm = bool(r.get("watermarked"))
            full = r.get("level") in FULL_AI
            if r.get("token_ids") and (full or not wm):
                sh.recent[(lang, wm, full)].append(r["token_ids"])
        sh.offset += end
        return sh

    def _verifier(self, cfg_path: Path):
        if cfg_path not in self._verifiers:
            add_synthid_to_path()
            from synthid_wm import SynthIDVerifier, load_config, load_tokenizer

            cfg = load_config(str(cfg_path), {"detection.min_scored": self.s["min_scored"]})
            if self._tok is None:
                self._tok = load_tokenizer(cfg.model.model_id)
            self._verifiers[cfg_path] = SynthIDVerifier(cfg.watermark, self._tok, cfg.detection)
        return self._verifiers[cfg_path]

    @staticmethod
    def _progress(log_path: Path) -> dict[str, Any]:
        out: dict[str, Any] = {"age_min": None}
        if not log_path.exists():
            return out
        out["age_min"] = (time.time() - log_path.stat().st_mtime) / 60
        with open(log_path, "rb") as f:
            f.seek(max(0, log_path.stat().st_size - 20000))
            tail = f.read().decode("utf-8", "replace").splitlines()
        for line in reversed(tail):
            if line.startswith("done:"):
                out["done"] = True
                break
            m = PROGRESS.match(line)
            if m:
                out.update(n=int(m[1]), total=int(m[2]), tok_s=int(m[3]), eta_min=float(m[4]))
                break
        for line in reversed(tail):
            if line.startswith("====="):
                break
            if "FAILED" in line or "Traceback" in line:
                out["error"] = line[:200]
        return out

    # ------------------------------------------------------------------ report

    def status(self) -> tuple[str, int]:
        s = self.s
        lines = [f"== watch {time.strftime('%Y-%m-%d %H:%M:%S %Z')} {self.dump}"]
        warns: list[str] = []
        gen = self.dump / "gen"
        streams = sorted(p.name for p in gen.iterdir() if p.is_dir()) if gen.is_dir() else []
        if not streams:
            lines.append("no generations yet")
        for stream in streams:
            files = sorted((p for p in (gen / stream).glob("a*.jsonl") if GEN_FILE.match(p.name)),
                           key=lambda p: tuple(int(x) for x in GEN_FILE.match(p.name).groups()))
            if not files:
                continue
            last_attempt = max(int(GEN_FILE.match(p.name)[1]) for p in files)
            by_lang: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
            recent: dict[tuple[str, bool, bool], list] = defaultdict(list)
            cfg_path = None
            for p in files:
                sh = self._read(p)
                for lang, st in sh.by_lang.items():
                    for k, v in st.items():
                        by_lang[lang][k] += v
                for key, dq in sh.recent.items():
                    recent[key].extend(dq)
                c = p.with_suffix(".config.yaml")
                if c.exists():
                    cfg_path = c
            # progress of the newest attempt's shards
            parts, tok_s, eta = [], 0, 0.0
            for p in files:
                a, i, n = (int(x) for x in GEN_FILE.match(p.name).groups())
                if a != last_attempt:
                    continue
                pr = self._progress(self.dump / "logs" / f"gen.{stream}.a{a}.s{i}of{n}.log")
                if pr.get("done"):
                    parts.append(f"s{i}: done")
                elif "n" in pr:
                    parts.append(f"s{i}: {pr['n']}/{pr['total']}")
                    tok_s += pr["tok_s"]
                    eta = max(eta, pr["eta_min"])
                    if pr["age_min"] is not None and pr["age_min"] > s["stall_min"]:
                        warns.append(f"{stream} shard {i}: log silent for {pr['age_min']:.0f} min")
                else:
                    parts.append(f"s{i}: starting")
                if pr.get("error"):
                    warns.append(f"{stream} shard {i}: {pr['error']}")
            n_rows = sum(int(st["rows"]) for st in by_lang.values())
            head = f"[{stream}] {n_rows} rows written (all attempts) | attempt {last_attempt} " + ", ".join(parts)
            if tok_s:
                head += f" | {tok_s} tok/s, ETA {eta / 60:.1f} h"
            lines.append(head)
            fails = sum(sum(1 for _ in open(f, encoding="utf-8")) for f in (gen / stream).glob("*.failures.jsonl"))
            if fails:
                warns.append(f"{stream}: {fails} failed batches (gen/{stream}/*.failures.jsonl)")
            for lang in sorted(by_lang):
                st = by_lang[lang]
                n = max(st["rows"], 1)
                row = (f"  {lang}: {int(st['rows'])} rows, eos {st['eos'] / n:.0%}, "
                       f"mean new tokens {st['new_tokens'] / n:.0f}")
                if st["empty"]:
                    row += f", empty {int(st['empty'])}"
                if lang == "hi":
                    row += f", not Devanagari {st['off_script'] / n:.0%}"
                    if st["rows"] >= s["min_rows"] and st["off_script"] / n > 0.1:
                        warns.append(f"{stream} hi: {st['off_script'] / n:.0%} of outputs are mostly not Devanagari")
                if st["rows"] >= s["min_rows"] and st["eos"] / n < s["min_eos"]:
                    warns.append(f"{stream} {lang}: only {st['eos'] / n:.0%} of outputs end at EOS")
                if st["think"]:
                    warns.append(f"{stream} {lang}: {int(st['think'])} outputs had <think> blocks")
                lines.append(row)
            if self.score and cfg_path is not None:
                try:
                    v = self._verifier(cfg_path)
                except Exception as e:  # noqa: BLE001  (no tokenizer offline, ...: progress still prints)
                    lines.append(f"  z check skipped: {type(e).__name__}: {str(e)[:120]}")
                    v = None
                for (lang, wm, full), ids_list in sorted(recent.items()) if v else []:
                    sample = ids_list[-int(s["sample"]):]
                    res = [v.verify_ids(ids) for ids in sample]
                    res = [r for r in res if r.n_scored >= s["min_scored"]]
                    if not res:
                        continue
                    zs = [r.z for r in res]
                    strength = statistics.mean(r.z / math.sqrt(r.n_scored) for r in res)
                    kind = "watermarked 100%-AI" if wm else "unwatermarked"
                    lines.append(f"  {lang} {kind}: newest {len(res)}: mean z {statistics.mean(zs):.2f}, "
                                 f"z>3 {sum(z > 3 for z in zs) / len(zs):.0%}, strength {strength:.3f}")
                    if len(res) >= s["min_rows"]:
                        if wm and full and statistics.mean(zs) < s["min_wm_z"]:
                            warns.append(f"{stream} {lang}: watermarked outputs score mean z {statistics.mean(zs):.2f}")
                        if not wm and statistics.mean(zs) > s["max_null_z"]:
                            warns.append(f"{stream} {lang}: unwatermarked outputs score mean z {statistics.mean(zs):.2f}")
        job_fail = self.dump / "failures" / "job.jsonl"
        if job_fail.exists() and job_fail.stat().st_size:
            n = sum(1 for _ in open(job_fail, encoding="utf-8"))
            warns.append(f"{n} failed stages (failures/job.jsonl)")
        lines += [f"WARN {w}" for w in warns]
        lines.append(f"STATUS {'WARN (' + str(len(warns)) + ')' if warns else 'OK'}")
        return "\n".join(lines), len(warns)

    def write(self) -> str:
        text, _ = self.status()
        (self.dump / "status.txt").write_text(text + "\n", encoding="utf-8")
        return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--no-score", action="store_true", help="skip the z check (no tokenizer needed)")
    ap.add_argument("--sample", type=int, help=f"rows scored per stream and language (default {DEFAULTS['sample']})")
    args = ap.parse_args(argv)
    settings = {"sample": args.sample} if args.sample else {}
    text, _ = Watcher(args.dump, settings, score=not args.no_score).status()
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
