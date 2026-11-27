"""Tests of analyze.py on a small hand-built dump in the run_study.py layout (no model, CPU only)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

import analyze
from conftest import REAL_TOKENIZER

HUMAN_EN = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
EDIT_EN = "alpha beta CHANGED delta epsilon zeta OTHER theta iota kappa"   # 2 of 10 words new -> f = 0.2
HUMAN_Z = [-1.5, -0.5, 0.0, 0.5, 1.5, 3.5]


def wj(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def score(r: dict, z: float, n_scored: int, n_masked: int = 0, thr: float = 3.0) -> dict:
    return {**{k: v for k, v in r.items() if k != "passage"}, "z": z, "n_scored": n_scored, "n_masked": n_masked,
            "flag": bool(n_scored >= 50 and z > thr), "decision": "x", "view": "passage"}


def make_dump(root: Path, with_gen: bool = True) -> Path:
    d = root / "dump"
    (d / "config").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"z_threshold": 3.0}), encoding="utf-8")
    (d / "config" / "run_config.yaml").write_text("watermark: {ngram_len: 5}\ndetection: {min_scored: 50}\n",
                                                  encoding="utf-8")
    trim_wm, sc_wm, gen_wm, raw = [], [], [], []
    # G: 10 prompt pairs, both languages; strength en 0.4, hi 0.2 exactly
    for i in range(10):
        for lang, s in (("en", 0.4), ("hi", 0.2)):
            for kind, cut, n in (("tokens", 400, 400), ("words", 200, 900 if lang == "hi" else 250)):
                r = {"id": f"{lang}-G-{i}", "pair_id": f"p{i}", "language": lang, "level": "G", "genre": "story",
                     "source": "t", "split": "main", "k": None, "cutoff_kind": kind, "cutoff": cut, "attempt": 0,
                     "watermarked": True, "accepted": True, "n_words": 200, "n_tokens": n, "passage": "x " * 10}
                trim_wm.append(r)
                sc_wm.append(score(r, s * math.sqrt(n), n, 0))
            raw.append({"id": f"{lang}-G-{i}", "language": lang, "level": "G", "attempt": 0, "watermarked": True,
                        "z": 5.0, "ids_z": 5.5 if i else 2.0, "ids_n_scored": 300, "flag": True,
                        "n_tokens_verified": 300, "ids_n_tokens": 300})
    # L2 edits: 4 docs per language, two cutoffs (the shorter one a prefix)
    for j in range(4):
        for lang in ("en", "hi"):
            rid = f"L2_polish-t-{j}-{lang}"
            gen_wm.append({"id": rid, "attempt": 0, "watermarked": True, "level": "L2_polish", "language": lang,
                           "human_passage": HUMAN_EN, "text": EDIT_EN})
            for cut, passage in ((10, EDIT_EN), (5, "alpha beta CHANGED delta epsilon")):
                r = {"id": rid, "pair_id": f"doc-{lang}-{j}", "language": lang, "level": "L2_polish",
                     "genre": "story" if j % 2 else "essay", "source": "t", "split": "main", "k": None,
                     "cutoff_kind": "words", "cutoff": cut, "attempt": 0, "watermarked": True, "accepted": True,
                     "n_words": cut, "n_tokens": 2 * cut, "passage": passage}
                trim_wm.append(r)
                sc_wm.append(score(r, 0.5, 60, 20))
    # C rows: AI shares from trim
    for j in range(3):
        r = {"id": f"C_continue_k50-t-{j}-en", "pair_id": f"doc-en-{j}", "language": "en", "level": "C_continue",
             "genre": "story", "source": "t", "split": "main", "k": 50, "cutoff_kind": "words", "cutoff": 10,
             "attempt": 0, "watermarked": True, "accepted": True, "n_words": 10, "n_tokens": 20,
             "ai_word_share": 0.5, "ai_token_share": 0.45, "passage": HUMAN_EN}
        trim_wm.append(r)
        sc_wm.append(score(r, 1.0, 60))
    # G_title (matched AI arm)
    for j in range(3):
        r = {"id": f"G_title-t-{j}-en", "pair_id": f"doc-en-{j}", "language": "en", "level": "G_title",
             "genre": "story", "source": "t", "split": "main", "k": None, "cutoff_kind": "words", "cutoff": 10,
             "attempt": 0, "watermarked": True, "accepted": True, "n_words": 10, "n_tokens": 20, "passage": EDIT_EN}
        trim_wm.append(r)
        sc_wm.append(score(r, 4.0, 100))
    wj(d / "trim" / "wm.jsonl", trim_wm)
    wj(d / "scores" / "wm.jsonl", sc_wm)
    wj(d / "scores" / "raw.wm.jsonl", raw)
    if with_gen:
        wj(d / "gen" / "wm" / "a0.s0of1.jsonl", gen_wm)
    # human pool: six known z values per language
    th, sh = [], []
    for lang in ("en", "hi"):
        for i, z in enumerate(HUMAN_Z):
            r = {"id": f"pool/{lang}{i}", "pair_id": f"pool/{lang}{i}", "language": lang, "level": "human",
                 "genre": "essay", "source": "t", "cutoff_kind": "tokens", "cutoff": 400, "attempt": None,
                 "trim_mode": "human", "accepted": True, "n_words": 100, "n_tokens": 400, "passage": HUMAN_EN}
            th.append(r)
            s = score(r, z, 380, 20)
            s["watermarked"] = "human"
            sh.append(s)
    wj(d / "trim" / "human.pool.jsonl", th)
    wj(d / "scores" / "human.pool.jsonl", sh)
    # unwatermarked control, G only
    tc, scc = [], []
    for i in range(4):
        r = {"id": f"en-G-{i}", "pair_id": f"p{i}", "language": "en", "level": "G", "genre": "story", "source": "t",
             "cutoff_kind": "tokens", "cutoff": 400, "attempt": 0, "watermarked": False, "accepted": True,
             "n_words": 300, "n_tokens": 400, "passage": "y"}
        tc.append(r)
        scc.append(score(r, 0.1 * i, 400))
    wj(d / "trim" / "ctrl.jsonl", tc)
    wj(d / "scores" / "ctrl.jsonl", scc)
    return d


def cfg(**over) -> dict:
    c = analyze.load_config()
    c["bootstrap"]["reps"] = 50
    c["figures"] = False
    c["length_trap"]["min_n"] = 2
    c.update(over)
    return c


@pytest.fixture()
def result(tmp_path):
    d = make_dump(tmp_path)
    return d, analyze.run(d, cfg(), tokenizer=None)


def by(rows, **kw):
    return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def test_copied_mask_and_normalisation():
    m = analyze.copied_mask(EDIT_EN.split(), HUMAN_EN.split())
    assert sum(m) == 8 and analyze.f_words_from_mask(m) == pytest.approx(0.2)
    # NFC and whitespace: a decomposed nukta and extra spaces are not edits
    a = analyze.normalize_text("क़  ख\nग")
    b = analyze.normalize_text("क़ ख ग")
    assert a == b
    # the windowed path covers every word
    long_ai = (EDIT_EN + " ") * 30
    long_h = (HUMAN_EN + " ") * 30
    mw = analyze.copied_mask(long_ai.split(), long_h.split(), max_words=50)
    assert len(mw) == 300 and 1 - sum(mw) / 300 == pytest.approx(0.2, abs=0.05)


def test_phase_mapping():
    assert analyze.phase_of("human.pool", "human") == "human"
    assert analyze.phase_of("ctrl", "G") == "control"
    assert analyze.phase_of("wm", "L3_paraphrase") == "edit"
    assert analyze.phase_of("wm", "C_continue") == "continue"
    assert analyze.phase_of("wm", "G_title") == "ai_matched"
    assert analyze.phase_of("wm", "O_outline") == "ai_matched"
    assert analyze.phase_of("wm", "G") == "ai_prompt"


def test_f_and_strength_per_phase(result):
    d, res = result
    rows = [json.loads(l) for l in (d / "analysis" / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
    ed = by(rows, level="L2_polish")
    assert len(ed) == 16 and all(r["f_words"] == pytest.approx(0.2) for r in ed)
    assert all(r["f_tokens"] is None for r in ed)            # no tokenizer
    assert all(r["mask_share"] == pytest.approx(0.25) for r in ed)
    assert all(r["f_words"] == 0.0 for r in by(rows, phase="human"))
    assert all(r["f_words"] == 1.0 for r in by(rows, phase="ai_prompt") + by(rows, phase="ai_matched"))
    c = by(rows, phase="continue")
    assert c and all(r["f_words"] == 0.5 and r["f_tokens"] == 0.45 for r in c)
    g = by(rows, phase="ai_prompt", language="en", cutoff=400)
    assert all(r["strength"] == pytest.approx(0.4) for r in g)
    assert all(r["view"] == "same_tokens" for r in g)
    assert all(r["phase"] == "control" for r in rows if r["stream"] == "ctrl")


def test_calibration(result):
    _, res = result
    cal = by(res["tables"]["calibration"], stream="human.pool", language="en")
    assert len(cal) == 1
    c = cal[0]
    assert c["n"] == 6 and c["mean_z"] == pytest.approx(sum(HUMAN_Z) / 6)
    assert c["flagged"] == 1 and c["fpr"] == pytest.approx(1 / 6)
    assert c["expected_fpr"] == pytest.approx(1 - analyze.norm_cdf(3.0))
    assert 0 < c["ks_distance"] < 1 and c["q999"] is None
    assert by(res["tables"]["calibration"], stream="ctrl")


def test_lang_gap_paired_and_primary(result):
    _, res = result
    gap = by(res["tables"]["lang_gap"], phase="ai_prompt", view="same_tokens", cutoff=400)
    assert len(gap) == 1
    g = gap[0]
    assert g["pairing"] == "paired" and g["n_shared"] == 10 and g["primary"] is True
    assert g["gap_hi_minus_en"] == pytest.approx(-0.2)
    assert g["ci_lo"] == pytest.approx(-0.2) and g["ci_hi"] == pytest.approx(-0.2)
    e = by(res["tables"]["lang_gap"], phase="edit", cutoff=10)
    assert e and e[0]["pairing"] == "independent"


def test_strength_table_and_bootstrap_determinism(tmp_path):
    d = make_dump(tmp_path)
    a = analyze.run(d, cfg(), out=tmp_path / "a")
    b = analyze.run(d, cfg(), out=tmp_path / "b")
    assert a["tables"]["strength"] == b["tables"]["strength"]
    s = by(a["tables"]["strength"], phase="edit", language="en", cutoff=10)[0]
    assert s["n"] == 4 and s["n_clusters"] == 4 and s["mean_f_words"] == pytest.approx(0.2)
    assert s["mean_strength"] == pytest.approx(0.5 / math.sqrt(60))
    assert s["genres"] == "essay,story" and s["gb_strength"] == pytest.approx(s["mean_strength"])
    gp = by(a["tables"]["strength"], phase="ai_prompt", language="hi", cutoff=400)[0]
    assert gp["detection_rate"] == 1.0                      # 0.2 * sqrt(400) = 4 > 3
    assert (tmp_path / "a" / "passages.csv").read_text(encoding="utf-8").startswith("stream,phase,id")


def test_f_curve_length_trap_retok(result):
    _, res = result
    fit = by(res["tables"]["f_fit"], language="en", view="same_words", cutoff=10)
    assert fit and fit[0]["c_matched_ai"] == pytest.approx(4.0 / 10.0)
    assert fit[0]["c_fit"] is not None
    bins = by(res["tables"]["f_curve"], language="en", cutoff=10)
    assert {b["f_lo"] for b in bins} == {0.2, 0.4}            # f 0.2 (L2) and 0.5 (C)
    lt = by(res["tables"]["length_trap"], phase="ai_prompt", language="hi", view="same_tokens")[0]
    assert lt["scored_tokens_needed"] == pytest.approx((3 / 0.2) ** 2)
    rt = by(res["tables"]["retok"], language="en")[0]
    assert rt["n"] == 10 and rt["share_flag_disagree"] == pytest.approx(0.1)


def test_missing_pieces(tmp_path):
    d = make_dump(tmp_path, with_gen=False)
    res = analyze.run(d, cfg())
    assert any("no human input" in n for n in res["notes"])
    empty = tmp_path / "empty"
    empty.mkdir()
    res = analyze.run(empty, cfg())
    assert res["n_passages"] == 0 and (empty / "analysis" / "analysis.md").exists()


@pytest.mark.skipif(not REAL_TOKENIZER.is_dir(), reason="real Qwen3 tokenizer not available")
def test_token_level_f(tmp_path):
    d = make_dump(tmp_path)
    res = analyze.run(d, cfg(), tokenizer=str(REAL_TOKENIZER))
    rows = [json.loads(l) for l in (d / "analysis" / "passages.jsonl").read_text(encoding="utf-8").splitlines()]
    ed = by(rows, level="L2_polish", cutoff=10)
    assert all(0.1 < r["f_tokens"] < 0.5 for r in ed)
    assert all(0 < r["f_ngram"] <= 1 for r in ed)
    assert res["ngram_len"] == 5


def test_cli_writes_every_file(tmp_path):
    d = make_dump(tmp_path)
    code = analyze.main([str(d), "--set", "bootstrap.reps=20", "--set", "figures=true"])
    assert code == 0
    out = d / "analysis"
    for name in ("passages.csv", "passages.jsonl", "calibration.csv", "strength.csv", "lang_gap.csv", "f_curve.csv",
                 "f_fit.csv", "length_trap.csv", "retok.csv", "analysis.md", "analysis_meta.json"):
        assert (out / name).exists(), name
    md = (out / "analysis.md").read_text(encoding="utf-8")
    assert md.index("## Primary") < md.index("## Calibration")
    try:
        import matplotlib  # noqa: F401
        assert (out / "figures" / "detection_same_tokens.png").exists()
    except ImportError:
        pass
    assert analyze.main([str(tmp_path / "nope")]) == 2
