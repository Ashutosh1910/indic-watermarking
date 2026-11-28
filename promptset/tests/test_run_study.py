"""End-to-end CPU test of the GPU job (run_study.py) with a tiny random Qwen3.

Builds real edit-level prompts (L2, C, OX -> O) from a small human pool, plus G rows, runs the whole job
into a dump, and checks: every stage's outputs, the per-token details against the scores, the loader's
joins and report, idempotent re-runs, resuming after a crash, and that a dump refuses other keys.
"""

from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import build_edit_prompts as bep
import build_human_pool
import load_dump
import run_study
from conftest import EN_TEXT, HI_TEXT, REPO, SCRIPTS

CONFIG = REPO / "promptset" / "config"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def g_rows() -> list[dict]:
    rows = []
    for i in range(6):
        lang = "en" if i % 2 == 0 else "hi"
        user = ("Write a short story about a lighthouse keeper." if lang == "en"
                else "एक दीपस्तंभ के रखवाले के बारे में एक छोटी कहानी लिखिए।") + " x" * i
        rows.append({
            "id": f"{lang}-G-t-{i}", "pair_id": f"t-{i}", "language": lang, "level": "G", "genre": "story",
            "source": "test", "split": "pilot" if i < 4 else "main", "length_tier": "short",
            "target_words": 40, "max_new_tokens": 96, "cutoffs": {"words": [10, 20], "tokens": [20, 40]},
            "messages": [{"role": "system", "content": "You are a writer."}, {"role": "user", "content": user}],
            "chat_template_kwargs": {"enable_thinking": False}, "generation": {"seed": 77 + i}, "watermark": True,
        })
    return rows


@pytest.fixture(scope="module")
def setup(tmp_path_factory, tiny_model_dir):
    d = tmp_path_factory.mktemp("study")
    # human docs: two per language, each a small variation of the conftest texts
    for lang, text in (("en", EN_TEXT), ("hi", HI_TEXT)):
        folder = d / "corpus" / lang
        folder.mkdir(parents=True)
        for j in range(2):
            sents = text.replace("\n", " ").split(". ")
            body = ". ".join(sents[j:] + sents[:j])
            (folder / f"doc{j}.txt").write_text(f"Title {j}\n\nAuthor\n\n{body}\n", encoding="utf-8")
        (folder / "README.md").write_text("not a document", encoding="utf-8")
    pool = d / "human_pool.jsonl"
    assert build_human_pool.main(["--out", str(pool), "--min-words", "30",
                                  "--dir", f"pool_en:en:{d / 'corpus' / 'en'}",
                                  "--dir", f"pool_hi:hi:{d / 'corpus' / 'hi'}"]) == 0

    # edit prompts with small cutoffs, from a copy of the edit config
    cfgdir = d / "cfg"
    shutil.copytree(CONFIG, cfgdir, ignore=shutil.ignore_patterns("study.yaml", "generation.yaml"))
    el = yaml.safe_load((cfgdir / "edit_levels.yaml").read_text(encoding="utf-8"))
    el["lengths"]["cutoffs"] = {"words": [10, 20], "tokens": [20, 40]}
    el["lengths"]["max_new_tokens_round"] = 64
    (cfgdir / "edit_levels.yaml").write_text(yaml.safe_dump(el, allow_unicode=True), encoding="utf-8")
    edit = d / "edit"
    bep.main(["build", "--config", str(cfgdir / "edit_levels.yaml"), "--jsonl", str(pool), "--tokenizer",
              str(tiny_model_dir), "--n-docs", "2", "--min-words", "30", "--split", "pilot", "--levels",
              "L2_polish", "C_continue", "O_outline", "--out", str(edit)])
    rows = read_jsonl(edit / "edit_prompts.jsonl")
    assert {r["level"] for r in rows} >= {"L2_polish", "C_continue", "OX_extract_outline"}
    for r in rows:   # keep the CPU run short
        r["max_new_tokens"] = min(r["max_new_tokens"], 128)
    write_jsonl(edit / "edit_prompts.jsonl", rows)
    G = write_jsonl(d / "G_prompts.jsonl", g_rows())

    gen = yaml.safe_load((CONFIG / "generation.yaml").read_text(encoding="utf-8"))
    gen["model"].update(model_id=str(tiny_model_dir), device="cpu", dtype="float32")
    gen["batching"] = {"batch_size": 8, "token_budget": 100000}
    gen["detection"]["min_scored"] = 5
    run_cfg = d / "run.yaml"
    run_cfg.write_text(yaml.safe_dump(gen), encoding="utf-8")

    study = yaml.safe_load((CONFIG / "study.yaml").read_text(encoding="utf-8"))
    # the run config keeps the public demo keys: the job must swap in private keys from keys.file (created here)
    # and build the human pool itself (build_inputs.human_pool) from the same corpus folders
    study.update(run_config=str(run_cfg), dump_dir=str(d / "dump"), gpus=0, filters="split=pilot",
                 prompts={"G": str(G), "edit": str(edit / "edit_prompts.jsonl")},
                 human={"edit_anchors": str(edit / "human_passages.jsonl"), "pool": str(d / "job_pool.jsonl"),
                        "absent": str(d / "nope.jsonl")},
                 human_cutoffs={"words": [10, 20], "tokens": [20, 40]},
                 keys={"file": str(d / "keys" / "private.yaml"), "allow_demo": False},
                 copy_results_to=str(d / "home_results"))
    study["build_inputs"]["human_pool"].update(
        dirs=[f"pool_en:en:{d / 'corpus' / 'en'}", f"pool_hi:hi:{d / 'corpus' / 'hi'}"], min_words=30)
    study["regen"]["max_attempts"] = 2
    study["detection"]["workers"] = 2
    sp = d / "study.yaml"
    sp.write_text(yaml.safe_dump(study, allow_unicode=True), encoding="utf-8")
    return {"dir": d, "study": sp, "dump": d / "dump", "run_cfg": run_cfg, "pool": pool}


def job(setup, *extra: str) -> int:
    return subprocess.run([sys.executable, str(SCRIPTS / "run_study.py"), "--study", str(setup["study"]), *extra],
                          cwd=str(REPO)).returncode


@pytest.fixture(scope="module")
def first_run(setup):
    assert job(setup, "--plan") == 0
    assert not list((setup["dump"] / "gen").glob("*/*.jsonl")), "--plan must not generate"
    assert job(setup) == 0
    return setup


def lines(p: Path) -> int:
    return sum(1 for l in p.read_text(encoding="utf-8").splitlines() if l.strip())


def test_outputs(first_run):
    dump = first_run["dump"]
    for rel in ("config/run_config.yaml", "config/study.yaml", "manifest.json", "job_runs.jsonl", "report.md",
                "gen/wm/a0.s0of1.jsonl", "gen/ox/a0.s0of1.jsonl", "gen/ctrl/a0.s0of1.jsonl",
                "trim/wm.jsonl", "trim/wm.cleaned.jsonl", "trim/ctrl.jsonl", "trim/human.pool.jsonl",
                "trim/human.edit_anchors.jsonl", "scores/wm.jsonl", "scores/raw.wm.jsonl", "scores/ctrl.jsonl",
                "scores/human.pool.jsonl", "tokens/wm.jsonl.gz", "tokens/human.pool.jsonl.gz",
                "prompts/O_prompts.jsonl", "failures/missing.wm.txt", "failures/dropped.wm.jsonl",
                "results/detection_rates.csv", "results/fpr_human.csv", "analysis/analysis.md",
                "analysis/passages.csv", "analysis/calibration.csv", "analysis/strength.csv", "analysis/lang_gap.csv"):
        assert (dump / rel).exists(), rel
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    assert man["status"] == "done", [s for s in man["stages"] if s["status"] != "done"]
    assert man["inputs"]["G"]["sha256"] and man["inputs"]["human.absent"]["exists"] is False
    assert {s["stage"] for s in man["stages"]} >= {"preflight", "gen:wm:a0", "trim:wm", "outline", "human:pool",
                                                   "score:wm", "score:raw.wm", "score:human.pool", "report",
                                                   "analysis", "inputs:edit", "inputs:human_pool"}
    gen = read_jsonl(dump / "gen" / "wm" / "a0.s0of1.jsonl")
    sel = (dump / "prompts" / "wm.selected_ids.txt").read_text(encoding="utf-8").split()
    assert {r["id"] for r in gen} == set(sel)                                # every selected row generated
    assert all(r["split"] == "pilot" and r["watermarked"] for r in gen)
    assert not any(r["level"] == "OX_extract_outline" for r in gen)
    assert all(r["language"] for r in gen) and {"en", "hi"} <= {r["language"] for r in gen}
    ox = read_jsonl(dump / "gen" / "ox" / "a0.s0of1.jsonl")
    assert ox and all(not r["watermarked"] and r["level"] == "OX_extract_outline" for r in ox)
    ctrl = read_jsonl(dump / "gen" / "ctrl" / "a0.s0of1.jsonl")
    assert ctrl and all(r["level"] == "G" and not r["watermarked"] for r in ctrl)
    for r in gen:   # what later analysis needs is logged per generation
        for k in ("text", "token_ids", "n_new_tokens", "finish_reason", "seed_effective", "batch_seed",
                  "config_sha", "prompt_tokens", "messages", "attempt"):
            assert k in r
    scores = read_jsonl(dump / "scores" / "wm.jsonl")
    assert scores and all(s["z_threshold"] == 3.0 and s["stream"] == "wm" and s["view"] == "passage"
                          for s in scores)
    assert all(s["flag"] == (s["z"] > 3.0 and s["n_scored"] >= 5) for s in scores)
    raw = read_jsonl(dump / "scores" / "raw.wm.jsonl")
    all_gen = [r for f in sorted((dump / "gen" / "wm").glob("a*.s*of*.jsonl")) if f.name.count(".") == 2
               for r in read_jsonl(f)]
    assert len(raw) == len(all_gen) > len(gen)          # every attempt, regenerations included
    assert {r["attempt"] for r in raw} == {0, 1, 2} and all("ids_z" in r for r in raw)
    hum = read_jsonl(dump / "scores" / "human.pool.jsonl")
    assert hum and all(s["watermarked"] == "human" for s in hum)
    assert {s["cutoff_kind"] for s in hum} == {"words", "tokens"}


def test_details_reproduce_scores(first_run):
    dump = first_run["dump"]
    for stream in ("wm", "human.pool"):
        scores = read_jsonl(dump / "scores" / f"{stream}.jsonl")
        details = read_jsonl(dump / "tokens" / f"{stream}.jsonl.gz")
        assert details
        for t in details:
            same = [s for s in scores if str(s["id"]) == str(t["id"]) and s.get("attempt") == t.get("attempt")
                    and s["cutoff_kind"] == t["cutoff_kind"] and s["cutoff"] == t["cutoff"]]
            assert len(same) == 1
            z, n = load_dump.prefix_z(t)
            assert n == same[0]["n_scored"]
            assert z == pytest.approx(same[0]["z"], abs=1e-9)
            assert len(t["g_bits"]) == len(t["mask"]) == max(0, len(t["token_ids"]) - (t["ngram_len"] - 1))


def test_loader(first_run):
    d = load_dump.Dump(first_run["dump"])
    res = d.results()
    assert res and all(r.get("accepted") is True and "passage" not in r for r in res)   # joined with trim rows
    assert {r["stream"] for r in res} >= {"wm", "human.pool", "human.edit_anchors"}   # ctrl: random model, may have none accepted
    assert {r["experiment_view"] for r in res} == {"same_words", "same_tokens"}
    rt = load_dump.rate_table(res, ["stream", "language"])
    assert all(0 <= r["rate"] <= 1 and r["n"] > 0 for r in rt)
    gens = list(d.generations(text=True))
    assert gens and all("text" in g and "token_ids" not in g for g in gens)
    f = d.failures()
    assert set(f) == {"job", "generate", "dropped", "missing"} and not f["job"] and not f["generate"]
    assert not f["missing"]
    df = d.results_df()
    assert len(df) == len(res)
    assert "Human text: false-positive rate" in (first_run["dump"] / "report.md").read_text(encoding="utf-8")


def test_rerun_is_idempotent(first_run):
    dump = first_run["dump"]
    before = {p: p.stat().st_mtime_ns for sub in ("gen", "trim", "scores", "tokens", "prompts")
              for p in (dump / sub).rglob("*") if p.is_file() and not p.name.endswith(".log")}
    assert job(first_run) == 0
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    work = [s for s in man["stages"] if s["status"] not in ("skipped",) and s["stage"] not in
            ("preflight", "report") and not s["stage"].startswith("missing:")]
    assert not work, work
    after = {p: p.stat().st_mtime_ns for p in before}
    assert before == after
    assert lines(dump / "job_runs.jsonl") == 2


def test_resume_after_crash(first_run):
    dump = first_run["dump"]
    f = dump / "gen" / "wm" / "a0.s0of1.jsonl"
    rows = f.read_text(encoding="utf-8").splitlines()
    n = len(rows)
    f.write_text("\n".join(rows[: n // 2]) + "\n" + rows[n // 2][:50], encoding="utf-8")   # crash mid-write
    assert job(first_run) == 0
    assert lines(f) == n
    assert len({json.loads(l)["id"] for l in f.read_text(encoding="utf-8").splitlines()}) == n
    man = json.loads((dump / "manifest.json").read_text(encoding="utf-8"))
    st: dict = {}
    for s in man["stages"]:
        st.setdefault(s["stage"], []).append(s)
    assert st["gen:wm:a0"][0]["status"] == "done" and st["gen:wm:a0"][0]["todo"] == n - n // 2
    assert st["trim:wm"][0]["status"] == "done"            # re-trimmed (a later final trim is then unchanged)
    assert st["score:wm"][0]["status"] == "done" and st["score:raw.wm"][0]["status"] == "done"


def test_more_gpus_later_skips_done_rows(first_run, tmp_path):
    # a new shard layout (here: the attempt-0 rows moved into a file of a 2-shard run) is still "done"
    dump = first_run["dump"]
    src = dump / "gen" / "wm" / "a0.s0of1.jsonl"
    import generate as G
    assert G.read_done_keys([src]) == G.read_done_keys([src, tmp_path / "missing.jsonl"])


def test_other_keys_refused(first_run, tmp_path):
    cfg = yaml.safe_load(first_run["run_cfg"].read_text(encoding="utf-8"))
    cfg["watermark"]["keys"] = [k + 1 for k in cfg["watermark"]["keys"]]
    other = tmp_path / "other.yaml"
    other.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert job(first_run, "--set", f"run_config={other}") != 0


def test_token_details_unit(tiny_model_dir):
    import synthid_wm
    from synthid_wm import SynthIDVerifier, load_tokenizer

    cfg = synthid_wm.load_config(None)
    tok = load_tokenizer(str(tiny_model_dir))
    v = SynthIDVerifier(cfg.watermark, tok, cfg.detection)
    ids = tok(EN_TEXT + " " + EN_TEXT, add_special_tokens=False)["input_ids"]   # repeated text -> masked contexts
    det = v.token_details(ids)
    res = v.verify_ids(ids)
    z, n = load_dump.prefix_z(det)
    assert n == res.n_scored and z == pytest.approx(res.z, abs=1e-9)
    assert sum(det["mask"]) == res.n_scored and len(det["mask"]) - sum(det["mask"]) == res.n_masked
    half = v.verify_ids(ids[:60])
    z60, n60 = load_dump.prefix_z(det, n_tokens=60)
    assert n60 == half.n_scored and z60 == pytest.approx(half.z, abs=1e-9)


def test_private_keys_and_inputs(first_run):
    """Demo keys in the run config -> private keys created once in keys.file (0600) and frozen in the dump;
    the human pool is built by the job (same documents as a hand-built pool); the edit file is kept as is."""
    from synthid_wm.config import BLOG_KEYS
    d, dump = first_run["dir"], first_run["dump"]
    kf = d / "keys" / "private.yaml"
    assert kf.exists() and (kf.stat().st_mode & 0o777) == 0o600
    keys = yaml.safe_load(kf.read_text(encoding="utf-8"))["watermark"]["keys"]
    frozen = yaml.safe_load((dump / "config" / "run_config.yaml").read_text(encoding="utf-8"))["watermark"]["keys"]
    assert keys == frozen and keys != list(BLOG_KEYS) and len(set(keys)) == 30
    first = json.loads((dump / "job_runs.jsonl").read_text(encoding="utf-8").splitlines()[0])  # the first real run
    st = {s["stage"]: s for s in first["stages"]}
    assert st["inputs:human_pool"]["status"] == "done"
    assert st["inputs:edit"]["status"] == "skipped"            # given by the test, not built by the job
    built = read_jsonl(d / "job_pool.jsonl")
    assert sorted(r["id"] for r in built) == sorted(r["id"] for r in read_jsonl(first_run["pool"]))
    assert (d / "job_pool.jsonl.job_build.json").exists()


def test_filters_from_the_command_line():
    """`--set filters='[split=pilot,main]'` is read by YAML as two items; the job joins them back."""
    assert run_study.normalize_filters(yaml.safe_load("[split=pilot,main, language=en]")) == \
        ["split=pilot,main", "language=en"]
    assert run_study.normalize_filters("split=pilot") == ["split=pilot"]
    with pytest.raises(SystemExit):
        run_study.normalize_filters(["main"])


def test_results_copied(first_run):
    """copy_results_to (Sharanga: $WM_RESULTS in $HOME) gets the small outputs, never keys or corpus text."""
    dest = first_run["dir"] / "home_results" / first_run["dump"].name
    for rel in ("report.md", "manifest.json", "results/detection_rates.csv", "analysis/analysis.md",
                "analysis/passages.csv"):
        assert (dest / rel).exists(), rel
    assert not (dest / "analysis" / "passages.jsonl").exists()
    assert not (dest / "config").exists() and not (dest / "trim").exists() and not (dest / "gen").exists()


def test_edit_subsample_keeps_same_docs_at_every_level():
    import run_study as R

    rows = []
    for lang in ("en", "hi"):
        for i in range(20):
            genre = "story" if i % 2 else "essay"
            split = "pilot" if i < 2 else "main"
            for level in ("L1_proofread", "L4_rewrite", "C_continue", "G_title", "OX_extract_outline"):
                for view in ("sw", "st"):
                    rows.append({"id": f"{level}-{lang}-{i}-{view}", "pair_id": f"{lang}/{i}", "language": lang,
                                 "genre": genre, "split": split, "level": level})
    spec = {"docs_per_language": 6, "levels": ["L1_proofread", "L4_rewrite", "C_continue"], "seed": 1}
    out = R.edit_subsample(rows, spec)
    assert out == R.edit_subsample(list(rows), spec)                       # deterministic
    for lang in ("en", "hi"):
        per_level = {lv: {r["pair_id"] for r in out if r["language"] == lang and r["level"] == lv}
                     for lv in spec["levels"]}
        docs = per_level["L1_proofread"]
        assert len(docs) == 6 and all(v == docs for v in per_level.values())
        assert {f"{lang}/0", f"{lang}/1"} <= docs                          # pilot docs always kept
        assert sum(int(d.split("/")[1]) % 2 for d in docs) == 3            # 3 story + 3 essay
        for lv in ("G_title", "OX_extract_outline"):                        # other levels untouched
            assert len({r["pair_id"] for r in out if r["language"] == lang and r["level"] == lv}) == 20
    assert R.edit_subsample(rows, {"docs_per_language": None, "levels": spec["levels"]}) == rows
