"""End-to-end CPU tests of generate.py -> trim.py -> verify_generations.py with a tiny random Qwen3.

The model is randomly initialised (2 layers, hidden 64), so its text is gibberish; what is tested is
the pipeline: outputs, resumability, sharding, attempts/seeds, batching, OOM splitting, and that the
watermark is measurable by synthid_wm's verifier.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

import generate
import trim
import verify_generations
from conftest import SCRIPTS
from generate import Request, cut_at_eos, plan_batches, run_with_oom_retry, select_rows, strip_think

N_ROWS = 12


def prompt_rows(n: int = N_ROWS) -> list[dict]:
    rows = []
    for i in range(n):
        lang = "en" if i % 2 == 0 else "hi"
        user = ("Write a short story about a lighthouse keeper." if lang == "en"
                else "एक दीपस्तंभ के रखवाले के बारे में एक छोटी कहानी लिखिए।") + " x" * (i % 4)
        rows.append({
            "id": f"{lang}-G-wp-{i}", "pair_id": f"wp-{i}", "language": lang,
            "level": "G" if i < 6 else "L2_polish", "genre": "story", "source": "writingprompts",
            "length_tier": "long", "target_words": 60, "max_new_tokens": 96 if i % 3 else 64,
            "min_new_tokens": None, "cutoffs": {"words": [5, 10], "tokens": [40, 60]},
            "messages": [{"role": "system", "content": "You are a writer."}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}, "generation": {"seed": 1000 + i},
            "watermark": i != 5, "split": "pilot" if i < 4 else "main", "extra_meta": {"i": i},
        })
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_gen(model_dir: Path, prompts: Path, out: Path, *extra: str) -> int:
    return generate.main(["--prompts", str(prompts), "--out", str(out), "--model", str(model_dir),
                          "--device", "cpu", "--dtype", "float32", "--batch-size", "4", *extra])


@pytest.fixture(scope="module")
def work(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("gen")


@pytest.fixture(scope="module")
def prompts(work) -> Path:
    return write_jsonl(work / "prompts.jsonl", prompt_rows())


@pytest.fixture(scope="module")
def base_out(tiny_model_dir, prompts, work) -> Path:
    out = work / "base" / "run.jsonl"
    assert run_gen(tiny_model_dir, prompts, out) == 0
    return out


@pytest.fixture(scope="module")
def control_out(tiny_model_dir, prompts, work) -> Path:
    out = work / "control" / "run_nowm.jsonl"
    assert run_gen(tiny_model_dir, prompts, out, "--no-watermark") == 0
    return out


def copy_run(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src.parent.iterdir():
        shutil.copy(p, dst_dir / p.name)
    return dst_dir / src.name


# --------------------------------------------------------------------------- outputs


def test_outputs_written(base_out, prompts):
    rows = read_jsonl(base_out)
    assert len(rows) == N_ROWS and len({r["id"] for r in rows}) == N_ROWS
    by_id = {r["id"]: r for r in rows}
    for src in prompt_rows():
        r = by_id[src["id"]]
        for k, v in src.items():
            assert r[k] == v, k                      # all input fields passed through, messages too
        assert r["finish_reason"] in ("eos", "length")
        assert r["n_new_tokens"] == len(r["token_ids"]) <= src["max_new_tokens"]
        if r["finish_reason"] == "length":
            assert r["n_new_tokens"] == src["max_new_tokens"]
        assert r["watermarked"] is (src["watermark"] is not False)
        assert r["attempt"] == 0 and r["seed_effective"] == src["generation"]["seed"]
        assert isinstance(r["text"], str) and r["had_think"] is False
        assert r["config_sha"] == rows[0]["config_sha"] and r["model_id"]
        assert r["seconds"] >= 0 and r["gen_time_utc"].endswith("+00:00")
    # batches never mix max_new_tokens or watermark flags, and respect --batch-size
    assert all(r["batch_size"] <= 4 for r in rows)


def test_saved_config_reloads_in_synthid(base_out):
    from synthid_wm import load_config

    cfg_path = base_out.parent / "run.config.yaml"
    cfg = load_config(cfg_path)
    assert generate.config_sha(cfg) == read_jsonl(base_out)[0]["config_sha"]
    assert cfg.generation.temperature == 0.7 and cfg.generation.top_k == 40 and cfg.generation.top_p == 1.0
    assert cfg.generation.enable_thinking is False
    assert (base_out.parent / "run.runs.jsonl").exists()


def test_chat_template_gets_enable_thinking_false(tiny_model_dir):
    from synthid_wm import load_tokenizer

    tok = load_tokenizer(str(tiny_model_dir))
    cfg, _ = generate.load_run_config(None)
    rows = prompt_rows(2)
    reqs = generate.build_requests(rows, tok, cfg, attempt=0, no_watermark=False)
    assert all(r.prompt_text.endswith("<think>\n\n</think>\n\n") for r in reqs)
    assert all(r.prompt_len == len(tok(r.prompt_text, add_special_tokens=False)["input_ids"]) for r in reqs)
    rows[0]["chat_template_kwargs"] = {"enable_thinking": True}
    reqs = generate.build_requests(rows, tok, cfg, attempt=0, no_watermark=False)
    assert not reqs[0].prompt_text.endswith("</think>\n\n")


# --------------------------------------------------------------------------- resumability


def test_second_run_does_nothing(tiny_model_dir, prompts, base_out, tmp_path):
    out = copy_run(base_out, tmp_path / "resume")
    before = out.read_bytes()
    assert run_gen(tiny_model_dir, prompts, out) == 0
    assert out.read_bytes() == before


def test_resume_after_crash_with_truncated_line(tiny_model_dir, prompts, base_out, tmp_path):
    out = copy_run(base_out, tmp_path / "crash")
    lines = base_out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:5]) + lines[5][:40], encoding="utf-8")   # crash mid-write
    assert run_gen(tiny_model_dir, prompts, out) == 0
    rows = read_jsonl(out)
    assert len(rows) == N_ROWS and len({r["id"] for r in rows}) == N_ROWS
    assert rows[:5] == read_jsonl(base_out)[:5]


def test_reproducible_with_same_batches(tiny_model_dir, prompts, base_out, tmp_path):
    out = tmp_path / "again" / "run.jsonl"
    assert run_gen(tiny_model_dir, prompts, out) == 0
    a = {r["id"]: r["token_ids"] for r in read_jsonl(base_out)}
    b = {r["id"]: r["token_ids"] for r in read_jsonl(out)}
    assert a == b


def test_config_change_is_refused(tiny_model_dir, prompts, base_out, tmp_path):
    out = copy_run(base_out, tmp_path / "cfgchange")
    raw = yaml.safe_load(generate.DEFAULT_CONFIG.read_text())
    raw["generation"]["temperature"] = 0.9
    cfg = tmp_path / "hot.yaml"
    cfg.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit):
        run_gen(tiny_model_dir, prompts, out, "--config", str(cfg))


# --------------------------------------------------------------------------- selection, shards, attempts


def test_shards_partition_rows(tiny_model_dir, prompts, tmp_path):
    ids = []
    for i in range(2):
        out = tmp_path / f"shard{i}" / "run.jsonl"
        assert run_gen(tiny_model_dir, prompts, out, "--shard", f"{i}/2") == 0
        ids.append({r["id"] for r in read_jsonl(out)})
    assert ids[0].isdisjoint(ids[1]) and ids[0] | ids[1] == {r["id"] for r in prompt_rows()}
    # larger n: still a partition, independent of filters
    rows = prompt_rows(200)
    parts = [{r["id"] for r in select_rows(rows, {}, None, (i, 5), None)} for i in range(5)]
    assert sum(map(len, parts)) == 200 and set().union(*parts) == {r["id"] for r in rows}
    with pytest.raises(ValueError):
        generate.parse_shard("2/2")


def test_filters_limit_only_ids(tiny_model_dir, prompts, tmp_path):
    out = tmp_path / "f" / "run.jsonl"
    assert run_gen(tiny_model_dir, prompts, out, "--filter", "language=hi", "--filter", "split=main",
                   "--limit", "3") == 0
    rows = read_jsonl(out)
    assert len(rows) == 3 and all(r["language"] == "hi" and r["split"] == "main" for r in rows)
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("en-G-wp-0\n# comment\nhi-G-wp-3\n")
    out2 = tmp_path / "f2" / "run.jsonl"
    assert run_gen(tiny_model_dir, prompts, out2, "--only-ids", str(ids_file)) == 0
    assert {r["id"] for r in read_jsonl(out2)} == {"en-G-wp-0", "hi-G-wp-3"}


def test_attempt_changes_seed(tiny_model_dir, prompts, base_out, tmp_path):
    out = copy_run(base_out, tmp_path / "attempt")
    assert run_gen(tiny_model_dir, prompts, out, "--attempt", "1") == 0
    rows = read_jsonl(out)
    assert len(rows) == 2 * N_ROWS
    a0 = {r["id"]: r for r in rows if r["attempt"] == 0}
    a1 = {r["id"]: r for r in rows if r["attempt"] == 1}
    assert a0.keys() == a1.keys()
    assert all(a0[i]["seed_effective"] != a1[i]["seed_effective"] for i in a0)
    assert sum(a0[i]["token_ids"] != a1[i]["token_ids"] for i in a0) >= N_ROWS - 1
    # re-running attempt 1 is a no-op too
    before = out.read_bytes()
    assert run_gen(tiny_model_dir, prompts, out, "--attempt", "1") == 0
    assert out.read_bytes() == before


def test_dry_run_writes_nothing(tiny_model_dir, prompts, tmp_path):
    out = tmp_path / "dry" / "run.jsonl"
    assert run_gen(tiny_model_dir, prompts, out, "--dry-run") == 0
    assert not out.exists() and not (tmp_path / "dry" / "run.config.yaml").exists()


def test_cli_runs_as_script(tiny_model_dir, prompts, tmp_path):
    cmd = [sys.executable, str(SCRIPTS / "generate.py"), "--prompts", str(prompts), "--out",
           str(tmp_path / "cli.jsonl"), "--model", str(tiny_model_dir), "--device", "cpu", "--limit", "2",
           "--dry-run"]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    assert "batches" in res.stderr


# --------------------------------------------------------------------------- batching helpers


def _req(i: int, plen: int, mnt: int, wm: bool = True, mn: int | None = None) -> Request:
    return Request(row={"id": str(i)}, id=str(i), prompt_text="", prompt_len=plen, max_new_tokens=mnt,
                   min_new_tokens=mn, watermark=wm, seed=i)


def test_plan_batches_groups_and_budget():
    reqs = ([_req(i, 100 + i, 2000) for i in range(10)] + [_req(100 + i, 50, 300) for i in range(7)]
            + [_req(200 + i, 80, 2000, wm=False) for i in range(3)] + [_req(300, 60, 300, mn=200)])
    batches = plan_batches(reqs, batch_size=4, token_budget=6500)
    assert sorted(r.id for b in batches for r in b) == sorted(r.id for r in reqs)
    for b in batches:
        assert len({(r.max_new_tokens, r.min_new_tokens, r.watermark) for r in b}) == 1
        assert len(b) <= 4
        longest = max(r.prompt_len for r in b)
        assert b[0].prompt_len == longest
        assert len(b) == 1 or len(b) * (longest + b[0].max_new_tokens) <= 6500
    assert [b[0].max_new_tokens for b in batches] == sorted((b[0].max_new_tokens for b in batches), reverse=True)
    # one row over budget still forms a batch of 1
    assert plan_batches([_req(0, 10_000, 2000)], 4, 100)[0][0].id == "0"


def test_oom_halves_batch_and_lowers_cap():
    calls: list[int] = []
    done: list[str] = []

    def fake(batch):
        calls.append(len(batch))
        if len(batch) > 2:
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        done.extend(r.id for r in batch)

    cap = [8]
    run_with_oom_retry([_req(i, 10, 10) for i in range(8)], fake, cap)
    assert sorted(done, key=int) == [str(i) for i in range(8)]
    assert calls == [8, 4, 2, 2, 4, 2, 2] and cap[0] == 2

    def other_error(batch):
        raise RuntimeError("some other failure")

    with pytest.raises(RuntimeError, match="other"):
        run_with_oom_retry([_req(0, 1, 1), _req(1, 1, 1)], other_error, [2])


def test_strip_think_and_cut_at_eos():
    assert strip_think("<think>\nplan\n</think>\n\nStory.") == ("Story.", True)
    assert strip_think("Story.") == ("Story.", False)
    assert strip_think("reasoning</think>\nStory.") == ("Story.", True)
    assert strip_think("<think>never closed") == ("", True)
    assert cut_at_eos([5, 6, 2, 0, 0], {2, 0}) == ([5, 6, 2], "eos")
    assert cut_at_eos([5, 6, 7], {2}) == ([5, 6, 7], "length")


def test_effective_seed():
    row = {"id": "x", "generation": {"seed": 42}}
    assert generate.effective_seed(row, 0) == 42
    s1, s2 = generate.effective_seed(row, 1), generate.effective_seed(row, 2)
    assert len({42, s1, s2}) == 3 and s1 == generate.effective_seed(row, 1)
    assert generate.effective_seed({"id": "y"}, 0) == generate.effective_seed({"id": "y"}, 0)


# --------------------------------------------------------------------------- watermark + downstream steps


def test_watermark_scores_higher_than_control(base_out, control_out, tmp_path):
    cfg = str(base_out.parent / "run.config.yaml")
    zs = {}
    for name, path in (("wm", base_out), ("ctl", control_out)):
        out = tmp_path / f"{name}.scores.jsonl"
        assert verify_generations.main(["--config", cfg, "--in", str(path), "--out", str(out), "--field", "text",
                                        "--score-ids"]) == 0
        rows = read_jsonl(out)
        assert len(rows) == N_ROWS
        assert all(math.isfinite(r["z"]) and math.isfinite(r["ids_z"]) for r in rows)
        zs[name] = rows
    wm_rows = [r for r in zs["wm"] if r["watermarked"] is True]
    ctl_rows = zs["ctl"] + [r for r in zs["wm"] if r["watermarked"] is False]
    wm_ids = statistics.fmean(r["ids_z"] for r in wm_rows)
    ctl_ids = statistics.fmean(r["ids_z"] for r in ctl_rows)
    wm_txt = statistics.fmean(r["z"] for r in wm_rows)
    ctl_txt = statistics.fmean(r["z"] for r in ctl_rows)
    print(f"\nmean z on sampled ids: wm {wm_ids:.2f} vs control {ctl_ids:.2f}; "
          f"on re-tokenised text: wm {wm_txt:.2f} vs control {ctl_txt:.2f}")
    # A random tiny model samples near-uniformly from the top-40, i.e. at high entropy, so the
    # watermark is strong on the sampled ids. Re-tokenised text loses part of it (byte-level drift).
    assert wm_ids > 6 and abs(ctl_ids) < 2.5
    assert wm_txt > ctl_txt + 1
    assert sum(r["ids_decision"] == "watermarked" for r in wm_rows) == len(wm_rows)
    assert sum(r["ids_decision"] == "watermarked" for r in ctl_rows) == 0
    summary = json.loads((tmp_path / "wm.scores.summary.json").read_text())
    assert sum(g["n"] for g in summary["groups"]) == N_ROWS


def test_trim_then_verify_pipeline(tiny_model_dir, base_out, tmp_path):
    out = tmp_path / "trim" / "run_trim.jsonl"
    assert trim.main(["gen", "--in", str(base_out), "--out", str(out), "--tokenizer", str(tiny_model_dir)]) == 0
    rows = read_jsonl(out)
    assert len(rows) == N_ROWS * 4                          # 2 word + 2 token cutoffs per row
    for r in rows:
        assert r["cutoff_kind"] in ("words", "tokens") and isinstance(r["accepted"], bool)
        if r["passage"]:
            assert r["n_tokens"] is not None and r["n_words"] == len(r["passage"].split())
        if r["accepted"]:
            unit = r["n_words"] if r["cutoff_kind"] == "words" else r["n_tokens"]
            assert 0.95 * r["cutoff"] <= unit <= r["cutoff"]
        else:
            assert r["reject_reason"] in ("too_short", "no_sentence_end", "script", "empty")
    for suffix in (".summary.md", ".summary.json", ".regen_ids.txt", ".cleaned.jsonl"):
        assert (tmp_path / "trim" / f"run_trim{suffix}").exists()
    scores = tmp_path / "scores.jsonl"
    assert verify_generations.main(["--config", str(base_out.parent / "run.config.yaml"), "--in", str(out),
                                    "--out", str(scores), "--include-rejected", "--min-scored", "5"]) == 0
    srows = read_jsonl(scores)
    assert len(srows) == len(rows) and all(math.isfinite(s["z"]) for s in srows)
