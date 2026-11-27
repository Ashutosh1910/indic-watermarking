"""Analysis of a run_study.py dump: per-token strength, measured edit share f, calibration and the language gap.

    python promptset/scripts/analyze.py dump/main --tokenizer Qwen/Qwen3-14B
    python promptset/scripts/analyze.py dump/main --set bootstrap.reps=200 --out /tmp/analysis

Reads the dump with load_dump.Dump (CPU only, deterministic for a given config) and writes into
``<dump>/analysis/`` (``--out`` to change):

    passages.csv / .jsonl   one row per scored passage: phase, view, cutoff, z, n_scored, strength, f_*, ...
    calibration.csv         human sets and the unwatermarked control: z mean / sd, KS distance, FPR
    strength.csv            detection rate and mean strength per cell, doc-clustered bootstrap CIs
    lang_gap.csv            Hindi minus English mean strength per cell (primary comparison marked)
    f_curve.csv             strength and detection by f bins, and the fit strength = c * f per language
    length_trap.csv         scored tokens (and words) a level needs for mean z to reach the threshold
    retok.csv               raw generations: z of the re-tokenised text vs z of the sampled ids
    analysis.md             the report; figures/*.png when matplotlib is installed

Definitions are in promptset/ANALYSIS.md. Summary:

* strength = z / sqrt(n_scored): watermark evidence per scored token. For watermarked text z grows about
  like sqrt(n_scored), so strength separates "how strong" from "how long".
* f = share of the passage the model actually wrote. Human 0; G, G_title and O_outline 1; C rows the AI share
  trim.py records; L1-L4 from a word alignment (difflib) of the passage with its human input after NFC and
  whitespace collapse. f_words counts words, f_tokens the tokens inside AI-written words, f_ngram the share of
  the passage's token n-grams (the detector's n-grams) that do not occur in the human input.
* Clusters for every bootstrap: pair_id (one source document for edit levels and human passages, one prompt
  pair for G). Cutoffs of one text are nested prefixes and levels share documents, so rows are not independent.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parents[1]
sys.path.insert(0, str(SCRIPTS))
from _common import log, md_table, stable_hash  # noqa: E402
from load_dump import GEN_FILE, Dump, _read, wilson, write_csv  # noqa: E402

DEFAULT_CONFIG = REPO / "promptset" / "config" / "analysis.yaml"
EDIT_LEVELS = ("L1_proofread", "L2_polish", "L3_paraphrase", "L4_rewrite")
MATCHED_LEVELS = ("G_title", "O_outline")
PHASES = ("human", "edit", "continue", "ai_matched", "ai_prompt", "control", "other")
VIEW = {"words": "same_words", "tokens": "same_tokens"}

# --------------------------------------------------------------------------- small helpers


def norm_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def fmean(vals: Iterable[Any]) -> float | None:
    v = [float(x) for x in vals if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)]
    return statistics.fmean(v) if v else None


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _set_dotted(d: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def load_config(path: str | Path | None = None, sets: Sequence[str] = ()) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    for s in sets:
        k, _, v = s.partition("=")
        _set_dotted(cfg, k.strip(), yaml.safe_load(v))
    return cfg


def normalize_text(s: str) -> str:
    """NFC and one space for every whitespace run, so Unicode encoding variants and layout do not count as edits."""
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def phase_of(stream: str, level: Any, control_streams: Sequence[str] = ("ctrl",)) -> str:
    s, lv = str(stream or ""), str(level or "")
    if s.startswith("human.") or lv == "human":
        return "human"
    if s in control_streams:
        return "control"
    if lv in EDIT_LEVELS:
        return "edit"
    if lv.startswith("C_continue"):
        return "continue"
    if lv in MATCHED_LEVELS:
        return "ai_matched"
    if lv == "G":
        return "ai_prompt"
    return "other"


def order_key(k: tuple) -> tuple:
    return tuple((0, v, "") if _num(v) else (1, 0, str(v)) for v in k)


# --------------------------------------------------------------------------- f: what the model wrote


def copied_mask(ai_words: Sequence[str], human_words: Sequence[str], max_words: int = 4000) -> list[bool]:
    """True for each AI word inside a difflib matching block with the human words (autojunk off).

    Above ``max_words`` both lists are cut into the same number of windows, aligned window by window
    (an approximation for very long texts that keeps the cost bounded; edits rarely move text far)."""
    n, m = len(ai_words), len(human_words)
    if n == 0:
        return []
    if m == 0:
        return [False] * n
    if max(n, m) > max_words:
        k = math.ceil(max(n, m) / max_words)
        out: list[bool] = []
        for i in range(k):
            a0, a1 = i * n // k, (i + 1) * n // k
            h0, h1 = i * m // k, (i + 1) * m // k
            out += copied_mask(ai_words[a0:a1], human_words[h0:h1], max_words)
        return out
    mask = [False] * n
    sm = difflib.SequenceMatcher(None, list(ai_words), list(human_words), autojunk=False)
    for a, _, size in sm.get_matching_blocks():
        for i in range(a, a + size):
            mask[i] = True
    return mask


def f_words_from_mask(mask: Sequence[bool]) -> float | None:
    return None if not mask else 1 - sum(mask) / len(mask)


class TokenTools:
    """Token-level f (needs a tokenizer): tokens inside AI-written words, and new n-grams."""

    def __init__(self, tokenizer: Any, ngram_len: int):
        self.tok = tokenizer
        self.n = ngram_len

    def ids(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    def word_token_counts(self, text: str) -> list[int]:
        """Tokens per whitespace word of ``text`` (normalised text), by character offsets."""
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        starts, pos = [], 0
        for w in text.split(" "):
            starts.append(pos)
            pos += len(w) + 1
        counts = [0] * len(starts)
        import bisect
        for a, b in enc["offset_mapping"]:
            if b <= a:
                continue
            mid = a if text[a:a + 1] != " " else a + 1
            j = bisect.bisect_right(starts, mid) - 1
            counts[max(0, min(j, len(counts) - 1))] += 1
        return counts

    def ngrams(self, ids: Sequence[int]) -> list[tuple[int, ...]]:
        return [tuple(ids[i - self.n + 1:i + 1]) for i in range(self.n - 1, len(ids))]


def generation_human_inputs(dump: Dump) -> dict[tuple[str, str, Any, Any], str]:
    """(stream, id, attempt, watermarked) -> human input passage, for rows that carry ``human_passage``."""
    out: dict[tuple[str, str, Any, Any], str] = {}
    gen_root = dump.path / "gen"
    if not gen_root.is_dir():
        return out
    for sdir in sorted(p for p in gen_root.iterdir() if p.is_dir()):
        for p in sorted(sdir.glob("a*.jsonl")):
            if not GEN_FILE.match(p.name):
                continue
            for r in _read(p):
                hp = r.get("human_passage")
                if hp and str(r.get("level")) in EDIT_LEVELS:
                    out[(sdir.name, str(r.get("id")), r.get("attempt"), r.get("watermarked"))] = hp
    return out


def attach_f(rows: list[dict[str, Any]], dump: Dump, tools: TokenTools | None, max_words: int) -> dict[str, int]:
    """Fill f_words / f_tokens / f_ngram on every passage row (in place). Returns counts for the report."""
    stats = defaultdict(int)
    human_in = generation_human_inputs(dump)
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        ph = r["phase"]
        if ph == "human":
            r.update(f_words=0.0, f_tokens=0.0, f_ngram=0.0)
        elif ph in ("ai_prompt", "ai_matched") or (ph == "control" and str(r.get("level")) in ("G",) + MATCHED_LEVELS):
            r.update(f_words=1.0, f_tokens=1.0, f_ngram=1.0)
        elif str(r.get("level", "")).startswith("C_continue"):
            fw, ft = r.get("ai_word_share"), r.get("ai_token_share")
            r.update(f_words=fw, f_tokens=ft, f_ngram=ft)
        elif str(r.get("level")) in EDIT_LEVELS:
            groups[(r["stream"], str(r["id"]), r.get("attempt"), r.get("watermarked"))].append(r)
        else:
            r.update(f_words=None, f_tokens=None, f_ngram=None)
    for key, grp in groups.items():
        human = human_in.get(key)
        if human is None:
            stats["edit_without_human_input"] += len(grp)
            for r in grp:
                r.update(f_words=None, f_tokens=None, f_ngram=None)
            continue
        hn = normalize_text(human)
        hw = hn.split(" ") if hn else []
        grp.sort(key=lambda r: len(r.get("passage") or ""), reverse=True)
        longest = normalize_text(grp[0].get("passage") or "")
        lw = longest.split(" ") if longest else []
        mask = copied_mask(lw, hw, max_words)
        stats["alignments"] += 1
        wtc = tools.word_token_counts(longest) if tools and longest else None
        h_ngrams = set(tools.ngrams(tools.ids(hn))) if tools else None
        long_ids = tools.ids(longest) if tools and longest else None
        for r in grp:
            pn = normalize_text(r.get("passage") or "")
            pw = pn.split(" ") if pn else []
            if pw and pw == lw[:len(pw)]:
                m, counts = mask[:len(pw)], (wtc[:len(pw)] if wtc else None)
                ids = (long_ids if len(pw) == len(lw) else tools.ids(pn)) if tools and pn else None
            else:
                stats["not_prefix"] += 1
                m = copied_mask(pw, hw, max_words)
                counts = tools.word_token_counts(pn) if tools and pn else None
                ids = tools.ids(pn) if tools and pn else None
            r["f_words"] = f_words_from_mask(m)
            if counts:
                tot = sum(counts)
                r["f_tokens"] = None if tot == 0 else sum(c for c, cp in zip(counts, m) if not cp) / tot
            else:
                r["f_tokens"] = None
            if tools and ids is not None:
                ng = tools.ngrams(ids)
                r["f_ngram"] = None if not ng else sum(g not in h_ngrams for g in ng) / len(ng)
            else:
                r["f_ngram"] = None
    return dict(stats)


# --------------------------------------------------------------------------- the tidy table

COLUMNS = ("stream", "phase", "id", "pair_id", "passage_id", "language", "level", "k", "genre", "source", "split",
           "cutoff_kind", "view", "cutoff", "n_words", "n_tokens", "n_scored", "n_masked", "mask_share", "z", "flag",
           "strength", "attempt", "watermarked", "f_words", "f_tokens", "f_ngram")


def tidy_rows(dump: Dump, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ctrl = tuple(cfg.get("control_streams") or ("ctrl",))
    rows = []
    for r in dump.results(text=True):
        ns, nm = r.get("n_scored"), r.get("n_masked")
        z = r.get("z")
        row = {k: r.get(k) for k in COLUMNS}
        row.update(passage=r.get("passage"), ai_word_share=r.get("ai_word_share"),
                   ai_token_share=r.get("ai_token_share"))
        row["view"] = VIEW.get(r.get("cutoff_kind"))
        row["phase"] = phase_of(r.get("stream"), r.get("level"), ctrl)
        row["pair_id"] = r.get("pair_id") or r.get("id")
        row["mask_share"] = (nm / (ns + nm)) if _num(ns) and _num(nm) and (ns + nm) > 0 else None
        row["strength"] = (z / math.sqrt(ns)) if _num(z) and _num(ns) and ns > 0 else None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- bootstrap


class Boot:
    """Cluster bootstrap of ratio-of-sums statistics (means and rates are ratios of sums over clusters)."""

    def __init__(self, reps: int, seed: int, ci: float = 0.95):
        import numpy as np
        self.np, self.reps, self.seed, self.ci = np, int(reps), int(seed), float(ci)

    def rng(self, *key: Any):
        return self.np.random.default_rng(stable_hash(self.seed, *[str(k) for k in key]))

    def ratio(self, num: dict[Any, float], den: dict[Any, float], key: tuple) -> tuple[float | None, float | None]:
        """CI of sum(num)/sum(den), resampling clusters (dict keys)."""
        np = self.np
        ks = [k for k in den if den[k] > 0]
        if len(ks) < 2 or self.reps <= 0:
            return None, None
        n = np.array([num.get(k, 0.0) for k in ks])
        d = np.array([den[k] for k in ks])
        idx = self.rng(*key).integers(0, len(ks), size=(self.reps, len(ks)))
        stat = n[idx].sum(1) / d[idx].sum(1)
        a = (1 - self.ci) / 2
        return float(np.quantile(stat, a)), float(np.quantile(stat, 1 - a))

    def stratified_balanced(self, strata: dict[Any, tuple[dict, dict]], key: tuple) -> tuple[float | None, float | None]:
        """CI of the equal-weight mean over strata of each stratum's ratio, resampling clusters within strata."""
        np = self.np
        parts = []
        for s, (num, den) in sorted(strata.items(), key=lambda kv: str(kv[0])):
            ks = [k for k in den if den[k] > 0]
            if len(ks) < 2:
                return None, None
            n = np.array([num.get(k, 0.0) for k in ks])
            d = np.array([den[k] for k in ks])
            idx = self.rng(*key, s).integers(0, len(ks), size=(self.reps, len(ks)))
            parts.append(n[idx].sum(1) / d[idx].sum(1))
        if not parts or self.reps <= 0:
            return None, None
        stat = np.mean(np.vstack(parts), axis=0)
        a = (1 - self.ci) / 2
        return float(np.quantile(stat, a)), float(np.quantile(stat, 1 - a))

    def diff_paired(self, a: dict[Any, tuple[float, float]], b: dict[Any, tuple[float, float]], key: tuple
                    ) -> tuple[float | None, float | None]:
        """CI of mean_a - mean_b, resampling clusters present in both (sums, counts per cluster)."""
        np = self.np
        ks = sorted(set(a) & set(b), key=str)
        if len(ks) < 2 or self.reps <= 0:
            return None, None
        an, ad = np.array([a[k][0] for k in ks]), np.array([a[k][1] for k in ks])
        bn, bd = np.array([b[k][0] for k in ks]), np.array([b[k][1] for k in ks])
        idx = self.rng(*key, "paired").integers(0, len(ks), size=(self.reps, len(ks)))
        stat = an[idx].sum(1) / ad[idx].sum(1) - bn[idx].sum(1) / bd[idx].sum(1)
        q = (1 - self.ci) / 2
        return float(np.quantile(stat, q)), float(np.quantile(stat, 1 - q))

    def diff_independent(self, a: dict[Any, tuple[float, float]], b: dict[Any, tuple[float, float]], key: tuple
                         ) -> tuple[float | None, float | None]:
        np = self.np
        if len(a) < 2 or len(b) < 2 or self.reps <= 0:
            return None, None

        def draws(d: dict[Any, tuple[float, float]], tag: str):
            ks = sorted(d, key=str)
            n, c = np.array([d[k][0] for k in ks]), np.array([d[k][1] for k in ks])
            idx = self.rng(*key, tag).integers(0, len(ks), size=(self.reps, len(ks)))
            return n[idx].sum(1) / c[idx].sum(1)

        stat = draws(a, "a") - draws(b, "b")
        q = (1 - self.ci) / 2
        return float(np.quantile(stat, q)), float(np.quantile(stat, 1 - q))


def cluster_sums(rows: Sequence[dict[str, Any]], value: Callable[[dict[str, Any]], Any]) -> dict[Any, tuple[float, float]]:
    out: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        v = value(r)
        if _num(v):
            c = out[r.get("pair_id")]
            c[0] += float(v)
            c[1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def _split(cs: dict[Any, tuple[float, float]]) -> tuple[dict, dict]:
    return {k: v[0] for k, v in cs.items()}, {k: v[1] for k, v in cs.items()}


def _flag(r: dict[str, Any]) -> float | None:
    f = r.get("flag")
    return None if f is None else float(bool(f))


# --------------------------------------------------------------------------- tables


def calibration_table(rows: Sequence[dict[str, Any]], thr: float) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r["phase"] in ("human", "control") and _num(r.get("z")) and r.get("n_scored"):
            groups[(r["stream"], r["language"], r["view"], r["cutoff"])].append(r)
    out = []
    for key in sorted(groups, key=order_key):
        g = groups[key]
        zs = sorted(float(r["z"]) for r in g)
        n = len(zs)
        ks = max(max((i + 1) / n - norm_cdf(z), norm_cdf(z) - i / n) for i, z in enumerate(zs))
        k = sum(bool(r.get("flag")) for r in g)
        lo, hi = wilson(k, n)
        out.append({"stream": key[0], "language": key[1], "view": key[2], "cutoff": key[3], "n": n,
                    "mean_z": statistics.fmean(zs), "sd_z": statistics.stdev(zs) if n > 1 else None,
                    "ks_distance": ks, "flagged": k, "fpr": k / n, "fpr_ci_lo": lo, "fpr_ci_hi": hi,
                    "expected_fpr": 1 - norm_cdf(thr),
                    "q999": zs[min(n - 1, int(math.ceil(0.999 * n)) - 1)] if n >= 1000 else None,
                    "mean_mask_share": fmean(r.get("mask_share") for r in g),
                    "mean_n_scored": fmean(r.get("n_scored") for r in g)})
    return out


CELL = ("stream", "phase", "language", "level", "k", "view", "cutoff")


def strength_table(rows: Sequence[dict[str, Any]], boot: Boot) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[tuple(r.get(k) for k in CELL)].append(r)
    out = []
    for key in sorted(groups, key=order_key):
        g = groups[key]
        st = cluster_sums(g, lambda r: r.get("strength"))
        fl = cluster_sums(g, _flag)
        s_num, s_den = _split(st)
        f_num, f_den = _split(fl)
        rec = dict(zip(CELL, key))
        rec.update(n=len(g), n_clusters=len({r.get("pair_id") for r in g}))
        rec["detection_rate"] = sum(f_num.values()) / sum(f_den.values()) if f_den else None
        rec["detection_ci_lo"], rec["detection_ci_hi"] = boot.ratio(f_num, f_den, key + ("rate",))
        rec["mean_strength"] = sum(s_num.values()) / sum(s_den.values()) if s_den else None
        rec["strength_ci_lo"], rec["strength_ci_hi"] = boot.ratio(s_num, s_den, key + ("strength",))
        rec.update(mean_z=fmean(r.get("z") for r in g), mean_f_words=fmean(r.get("f_words") for r in g),
                   mean_f_tokens=fmean(r.get("f_tokens") for r in g), mean_f_ngram=fmean(r.get("f_ngram") for r in g),
                   mean_mask_share=fmean(r.get("mask_share") for r in g),
                   mean_n_scored=fmean(r.get("n_scored") for r in g), mean_n_words=fmean(r.get("n_words") for r in g),
                   mean_n_tokens=fmean(r.get("n_tokens") for r in g))
        # genre-balanced: equal weight per genre present in the cell
        by_g: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for r in g:
            by_g[r.get("genre")].append(r)
        rec["genres"] = ",".join(sorted(str(x) for x in by_g))
        if len(by_g) > 1:
            gs = {gn: _split(cluster_sums(rs, lambda r: r.get("strength"))) for gn, rs in by_g.items()}
            gf = {gn: _split(cluster_sums(rs, _flag)) for gn, rs in by_g.items()}
            rec["gb_strength"] = fmean(sum(n.values()) / sum(d.values()) for n, d in gs.values() if d)
            rec["gb_strength_ci_lo"], rec["gb_strength_ci_hi"] = boot.stratified_balanced(gs, key + ("gbs",))
            rec["gb_detection_rate"] = fmean(sum(n.values()) / sum(d.values()) for n, d in gf.values() if d)
            rec["gb_detection_ci_lo"], rec["gb_detection_ci_hi"] = boot.stratified_balanced(gf, key + ("gbr",))
        else:
            rec.update(gb_strength=rec["mean_strength"], gb_strength_ci_lo=rec["strength_ci_lo"],
                       gb_strength_ci_hi=rec["strength_ci_hi"], gb_detection_rate=rec["detection_rate"],
                       gb_detection_ci_lo=rec["detection_ci_lo"], gb_detection_ci_hi=rec["detection_ci_hi"])
        out.append(rec)
    return out


GAP_CELL = ("stream", "phase", "level", "k", "view", "cutoff")


def lang_gap_table(rows: Sequence[dict[str, Any]], boot: Boot, primary: dict[str, Any]) -> list[dict[str, Any]]:
    groups: dict[tuple, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("language") in ("hi", "en"):
            key = tuple(r.get(k) for k in GAP_CELL)     # human sets: languages compared within each set
            groups[key][r["language"]].append(r)
    out = []
    for key in sorted(groups, key=order_key):
        g = groups[key]
        if not g.get("hi") or not g.get("en"):
            continue
        hi = cluster_sums(g["hi"], lambda r: r.get("strength"))
        en = cluster_sums(g["en"], lambda r: r.get("strength"))
        if not hi or not en:
            continue
        m_hi = sum(v[0] for v in hi.values()) / sum(v[1] for v in hi.values())
        m_en = sum(v[0] for v in en.values()) / sum(v[1] for v in en.values())
        shared = set(hi) & set(en)
        paired = len(shared) >= 0.5 * min(len(hi), len(en)) and len(shared) >= 2
        if paired:
            lo, up = boot.diff_paired(hi, en, key)
            hs = {k: hi[k] for k in shared}
            es = {k: en[k] for k in shared}
            diff = (sum(v[0] for v in hs.values()) / sum(v[1] for v in hs.values())
                    - sum(v[0] for v in es.values()) / sum(v[1] for v in es.values()))
        else:
            lo, up = boot.diff_independent(hi, en, key)
            diff = m_hi - m_en
        rec = dict(zip(GAP_CELL, key))
        rec.update(strength_hi=m_hi, strength_en=m_en, gap_hi_minus_en=diff, ci_lo=lo, ci_hi=up,
                   pairing="paired" if paired else "independent", n_clusters_hi=len(hi), n_clusters_en=len(en),
                   n_shared=len(shared),
                   ratio_hi_over_en=(m_hi / m_en) if m_en else None,
                   primary=bool(rec["view"] == primary.get("view") and rec["cutoff"] == primary.get("cutoff")))
        out.append(rec)
    return out


def f_curve_table(rows: Sequence[dict[str, Any]], boot: Boot, bins: Sequence[float]
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(binned rows, fit rows). Edit + continue phases of watermarked streams; ai_matched gives c at f = 1."""
    bins = sorted(float(b) for b in bins)
    ed = [r for r in rows if r["phase"] in ("edit", "continue") and _num(r.get("f_words")) and _num(r.get("strength"))]
    binned: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in ed:
        f = float(r["f_words"])
        for lo, hi in zip(bins[:-1], bins[1:]):
            if lo <= f + 1e-9 < hi:   # 1 - 8/10 is 0.19999... in floating point
                binned[(r["stream"], r["language"], r["view"], r["cutoff"], lo, hi)].append(r)
                break
    out = []
    for key in sorted(binned, key=order_key):
        g = binned[key]
        k = sum(bool(r.get("flag")) for r in g)
        lo, hi = wilson(k, len(g))
        out.append({"stream": key[0], "language": key[1], "view": key[2], "cutoff": key[3], "f_lo": key[4],
                    "f_hi": key[5], "n": len(g), "mean_f_words": fmean(r["f_words"] for r in g),
                    "mean_strength": fmean(r["strength"] for r in g), "detection_rate": k / len(g),
                    "detection_ci_lo": lo, "detection_ci_hi": hi})
    fits = []
    cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in ed:
        cells[(r["stream"], r["language"], r["view"], r["cutoff"])].append(r)
    matched: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        if r["phase"] == "ai_matched" and _num(r.get("strength")):
            matched[(r["stream"], r["language"], r["view"], r["cutoff"])].append(float(r["strength"]))
    for key in sorted(cells, key=order_key):
        g = cells[key]
        num: dict[Any, float] = defaultdict(float)
        den: dict[Any, float] = defaultdict(float)
        for r in g:
            f, s = float(r["f_words"]), float(r["strength"])
            num[r.get("pair_id")] += f * s
            den[r.get("pair_id")] += f * f
        c = sum(num.values()) / sum(den.values()) if sum(den.values()) > 0 else None
        lo, hi = boot.ratio(dict(num), dict(den), key + ("c",))
        m = matched.get(key)
        fits.append({"stream": key[0], "language": key[1], "view": key[2], "cutoff": key[3], "n": len(g),
                     "c_fit": c, "c_ci_lo": lo, "c_ci_hi": hi,
                     "c_matched_ai": statistics.fmean(m) if m else None, "n_matched_ai": len(m) if m else 0})
    return out, fits


def length_trap_table(rows: Sequence[dict[str, Any]], thr: float, min_n: int) -> list[dict[str, Any]]:
    cells: dict[tuple, dict[Any, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["phase"] in ("edit", "continue", "ai_matched", "ai_prompt") and _num(r.get("strength")):
            cells[(r["stream"], r["phase"], r["language"], r["level"], r.get("k"), r["view"])][r["cutoff"]].append(r)
    out = []
    for key in sorted(cells, key=order_key):
        cuts = [c for c in sorted(cells[key], key=lambda x: (0, x) if _num(x) else (1, 0)) if len(cells[key][c]) >= min_n]
        if not cuts:
            continue
        c = cuts[-1]
        g = cells[key][c]
        s = statistics.median(float(r["strength"]) for r in g)
        ms = fmean(r.get("mask_share") for r in g) or 0.0
        tpw = fmean((r["n_tokens"] / r["n_words"]) for r in g if _num(r.get("n_tokens")) and r.get("n_words"))
        need_scored = (thr / s) ** 2 if s > 0 else None
        need_tokens = need_scored / (1 - ms) if need_scored is not None and ms < 1 else None
        out.append({"stream": key[0], "phase": key[1], "language": key[2], "level": key[3], "k": key[4],
                    "view": key[5], "cutoff_used": c, "n": len(g), "median_strength": s, "mean_mask_share": ms,
                    "tokens_per_word": tpw, "scored_tokens_needed": need_scored, "tokens_needed": need_tokens,
                    "words_needed": need_tokens / tpw if need_tokens is not None and tpw else None,
                    "detected_at_cutoff_used": s * math.sqrt(fmean(r["n_scored"] for r in g) or 0) > thr})
    return out


def retok_table(dump: Dump, thr: float, min_scored: int) -> list[dict[str, Any]]:
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in dump.scores("raw"):
        if _num(r.get("z")) and _num(r.get("ids_z")):
            groups[(r.get("stream"), r.get("language"), r.get("level"))].append(r)
    out = []
    for key in sorted(groups, key=order_key):
        g = groups[key]
        tf = [bool(r.get("flag")) if r.get("flag") is not None else (r["z"] > thr) for r in g]
        idf = [r["ids_z"] > thr and (r.get("ids_n_scored") or 0) >= min_scored for r in g]
        out.append({"stream": key[0], "language": key[1], "level": key[2], "n": len(g),
                    "mean_z_text": fmean(r["z"] for r in g), "mean_z_ids": fmean(r["ids_z"] for r in g),
                    "mean_diff_text_minus_ids": fmean(r["z"] - r["ids_z"] for r in g),
                    "share_flag_disagree": sum(a != b for a, b in zip(tf, idf)) / len(g),
                    "mean_tokens_text": fmean(r.get("n_tokens_verified") for r in g),
                    "mean_tokens_ids": fmean(r.get("ids_n_tokens") for r in g)})
    return out


# --------------------------------------------------------------------------- figures


def figures(rows: Sequence[dict[str, Any]], strength: Sequence[dict[str, Any]], fig_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for view in ("same_words", "same_tokens"):
        cells = [s for s in strength if s["view"] == view and s["phase"] != "human" and _num(s.get("detection_rate"))]
        if not cells:
            continue
        phases = sorted({s["phase"] for s in cells})
        fig, axes = plt.subplots(1, len(phases), figsize=(4 * len(phases), 3.4), squeeze=False)
        for ax, ph in zip(axes[0], phases):
            series: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
            for s in cells:
                if s["phase"] == ph and _num(s["cutoff"]):
                    series[(s["stream"], s["language"], s["level"], s.get("k"))].append((s["cutoff"], s["detection_rate"]))
            for (st, lang, lv, k), pts in sorted(series.items(), key=lambda kv: order_key(kv[0])):
                pts.sort()
                lab = f"{lang} {lv}{'' if k is None else f' k{k}'}{'' if st == 'wm' else f' ({st})'}"
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ls="-" if lang == "en" else "--", label=lab)
            ax.set_title(ph)
            ax.set_xlabel("cutoff (words)" if view == "same_words" else "cutoff (tokens)")
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=6)
        axes[0][0].set_ylabel("detection rate")
        fig.tight_layout()
        p = fig_dir / f"detection_{view}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p.name)
    pts = [r for r in rows if r["phase"] in ("edit", "continue") and _num(r.get("f_words")) and _num(r.get("strength"))]
    if pts:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for lang, mk in (("en", "o"), ("hi", "^")):
            sel = [r for r in pts if r["language"] == lang]
            if sel:
                ax.scatter([r["f_words"] for r in sel], [r["strength"] for r in sel], s=6, alpha=0.4, marker=mk, label=lang)
        ax.set_xlabel("f (share of words the model wrote)")
        ax.set_ylabel("strength = z / sqrt(n_scored)")
        ax.legend()
        fig.tight_layout()
        p = fig_dir / "strength_vs_f.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p.name)
    hum = [r for r in rows if r["phase"] == "human" and _num(r.get("z"))]
    if hum:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for lang in ("en", "hi"):
            zs = [float(r["z"]) for r in hum if r["language"] == lang]
            if zs:
                ax.hist(zs, bins=40, density=True, histtype="step", label=f"{lang} (n={len(zs)})")
        xs = [x / 20 for x in range(-100, 101)]
        ax.plot(xs, [math.exp(-x * x / 2) / math.sqrt(2 * math.pi) for x in xs], "k:", label="N(0,1)")
        ax.set_xlabel("z (human text)")
        ax.legend()
        fig.tight_layout()
        p = fig_dir / "human_z.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(p.name)
    return made


# --------------------------------------------------------------------------- report


def _md(rows: Sequence[dict[str, Any]], cols: Sequence[str], limit: int = 200) -> str:
    if not rows:
        return "(none)"
    body = [[r.get(c) for c in cols] for r in rows[:limit]]
    more = f"\n\n({len(rows) - limit} more rows in the CSV)" if len(rows) > limit else ""
    return md_table(list(cols), body) + more


def report_md(res: dict[str, Any], cfg: dict[str, Any], thr: float, notes: list[str]) -> str:
    prim = cfg.get("primary") or {}
    L = [f"# Analysis: {res['dump']}", "",
         f"Flag: z > {thr}. strength = z / sqrt(n_scored). f = share of the passage the model wrote. "
         f"CIs: {int(100 * float(cfg['bootstrap'].get('ci', 0.95)))}% cluster bootstrap over pair_id, "
         f"{cfg['bootstrap'].get('reps')} resamples. Definitions: promptset/ANALYSIS.md.", ""]
    gap = res["tables"]["lang_gap"]
    L += [f"## Primary: Hindi minus English strength ({prim.get('view')}, cutoff {prim.get('cutoff')})", "",
          _md([g for g in gap if g.get("primary")],
              ["stream", "phase", "level", "k", "strength_hi", "strength_en", "gap_hi_minus_en", "ci_lo", "ci_hi",
               "pairing"]), ""]
    L += ["## Calibration (human text and unwatermarked control)", "",
          "Under the null z is N(0,1): mean near 0, sd near 1, small KS distance. FPR counts are small, so read "
          "the z distribution first.", "",
          _md(res["tables"]["calibration"], ["stream", "language", "view", "cutoff", "n", "mean_z", "sd_z",
                                             "ks_distance", "flagged", "fpr", "fpr_ci_hi", "expected_fpr",
                                             "mean_mask_share"]), ""]
    L += ["## Strength and detection per cell", "",
          _md(res["tables"]["strength"], ["stream", "phase", "language", "level", "k", "view", "cutoff", "n",
                                          "n_clusters", "detection_rate", "detection_ci_lo", "detection_ci_hi",
                                          "mean_strength", "strength_ci_lo", "strength_ci_hi", "gb_strength",
                                          "mean_f_words", "mean_mask_share"]), ""]
    L += ["## Language gap, every cell", "",
          _md(gap, ["stream", "phase", "level", "k", "view", "cutoff", "strength_hi", "strength_en",
                    "gap_hi_minus_en", "ci_lo", "ci_hi", "pairing", "primary"]), ""]
    L += ["## Strength at equal f (edit and continue)", "", "Fit strength = c * f through the origin; c_matched_ai "
          "is the mean strength of the matched 100% AI arm (G_title, O_outline) in the same cell.", "",
          _md(res["tables"]["f_fit"], ["stream", "language", "view", "cutoff", "n", "c_fit", "c_ci_lo", "c_ci_hi",
                                       "c_matched_ai"]), "",
          _md(res["tables"]["f_curve"], ["stream", "language", "view", "cutoff", "f_lo", "f_hi", "n", "mean_f_words",
                                         "mean_strength", "detection_rate"]), ""]
    L += ["## Length trap", "", "Tokens a level needs before its median text reaches the threshold: "
          "(threshold / strength)^2 scored tokens, divided by (1 - mask share), and in words via tokens per word.", "",
          _md(res["tables"]["length_trap"], ["stream", "phase", "language", "level", "k", "view", "cutoff_used", "n",
                                             "median_strength", "tokens_needed", "words_needed"]), ""]
    L += ["## Retokenisation (full raw generations)", "",
          _md(res["tables"]["retok"], ["stream", "language", "level", "n", "mean_z_text", "mean_z_ids",
                                       "mean_diff_text_minus_ids", "share_flag_disagree"]), ""]
    if res.get("figures"):
        L += ["## Figures", ""] + [f"![{f}](figures/{f})" for f in res["figures"]] + [""]
    L += ["## Caveats", "",
          "* Cutoffs of one text are nested prefixes, and edit levels share source documents. Rows are not "
          "independent; every CI resamples whole pair_id clusters.",
          "* Hindi same-tokens passages are the opening of a text (1,500 tokens is about 330 Hindi words), while "
          "English ones cover most of it.",
          "* f_words aligns words after NFC and whitespace collapse. f_tokens counts tokens inside AI-written words. "
          "f_ngram uses the detector's n-gram length. Very long texts are aligned in windows (ANALYSIS.md).",
          "* strength assumes z grows like sqrt(n_scored), which holds when the per-token signal is steady along "
          "the text.", ""]
    if notes:
        L += ["## Notes", ""] + [f"* {n}" for n in notes] + [""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main


def _run_config(dump: Dump) -> dict[str, Any]:
    p = dump.path / "config" / "run_config.yaml"
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


def load_tokenizer(spec: Any):
    if spec is None or spec is False or str(spec).lower() in ("", "none"):
        return None
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(spec))


def run(dump_path: str | Path, cfg: dict[str, Any] | None = None, tokenizer: Any = None,
        out: str | Path | None = None) -> dict[str, Any]:
    """Analyse one dump. ``tokenizer`` is a tokenizer object, a path / HF id, or None (no token-level f)."""
    cfg = cfg if cfg is not None else load_config()
    cfg.setdefault("bootstrap", {})
    dump = Dump(dump_path)
    out_dir = Path(out) if out else dump.path / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    man = dump.manifest() if (dump.path / "manifest.json").exists() else {}
    rc = _run_config(dump)
    thr = float(cfg.get("z_threshold") or man.get("z_threshold") or 3.0)
    min_scored = int(cfg.get("min_scored") or (rc.get("detection") or {}).get("min_scored") or 50)
    ngram = int((cfg.get("f") or {}).get("ngram_len") or (rc.get("watermark") or {}).get("ngram_len") or 5)
    tok = load_tokenizer(tokenizer) if isinstance(tokenizer, (str, Path)) or tokenizer is None else tokenizer
    if tok is None:
        notes.append("no tokenizer given: f_tokens and f_ngram are empty for L1-L4 rows")
    tools = TokenTools(tok, ngram) if tok is not None else None

    rows = tidy_rows(dump, cfg) if (dump.path / "scores").is_dir() else []
    if not rows:
        notes.append("no scored passages in the dump (scores/ missing or empty)")
    fstats = attach_f(rows, dump, tools, int((cfg.get("f") or {}).get("max_align_words", 4000)))
    if fstats.get("edit_without_human_input"):
        notes.append(f"{fstats['edit_without_human_input']} edit passages had no human input in gen/ (f empty)")
    if fstats.get("not_prefix"):
        notes.append(f"{fstats['not_prefix']} edit passages were not a prefix of their longest passage (aligned directly)")
    present = {r["phase"] for r in rows}
    for ph in ("human", "edit", "continue", "ai_matched", "ai_prompt", "control"):
        if ph not in present:
            notes.append(f"phase {ph}: no passages in this dump")

    b = cfg["bootstrap"]
    boot = Boot(int(b.get("reps", 1000)), int(b.get("seed", 0)), float(b.get("ci", 0.95)))
    tables = {
        "calibration": calibration_table(rows, thr),
        "strength": strength_table(rows, boot),
        "lang_gap": lang_gap_table(rows, boot, cfg.get("primary") or {}),
        "length_trap": length_trap_table(rows, thr, int((cfg.get("length_trap") or {}).get("min_n", 10))),
        "retok": retok_table(dump, thr, min_scored) if (dump.path / "scores").is_dir() else [],
    }
    tables["f_curve"], tables["f_fit"] = f_curve_table(rows, boot, (cfg.get("f") or {}).get("bins")
                                                       or [0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0001])
    tidy = [{k: r.get(k) for k in COLUMNS} for r in rows]
    write_csv(out_dir / "passages.csv", tidy)
    with open(out_dir / "passages.jsonl", "w", encoding="utf-8") as f:
        for r in tidy:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for name, t in tables.items():
        write_csv(out_dir / f"{name}.csv", t)
    figs = figures(rows, tables["strength"], out_dir / "figures") if cfg.get("figures", True) and rows else []
    if cfg.get("figures", True) and rows and not figs:
        notes.append("figures skipped (matplotlib not installed)")
    res = {"dump": str(dump.path), "out": str(out_dir), "n_passages": len(rows), "z_threshold": thr,
           "ngram_len": ngram, "f_stats": fstats, "figures": figs, "notes": notes,
           "tables": tables, "config": cfg}
    (out_dir / "analysis.md").write_text(report_md(res, cfg, thr, notes), encoding="utf-8")
    meta = {k: v for k, v in res.items() if k != "tables"}
    meta["rows_per_table"] = {k: len(v) for k, v in tables.items()}
    (out_dir / "analysis_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=str),
                                                encoding="utf-8")
    return res


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--tokenizer", help="tokenizer dir or HF id (the job's tokenizer); omit for word-level f only")
    ap.add_argument("--out", help="output directory (default <dump>/analysis)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    args = ap.parse_args(argv)
    if not Path(args.dump).is_dir():
        log(f"dump not found: {args.dump}")
        return 2
    cfg = load_config(args.config, args.set)
    res = run(args.dump, cfg, args.tokenizer, args.out)
    log(f"analysis: {res['n_passages']} passages -> {res['out']} "
        f"({', '.join(f'{k} {len(v)}' for k, v in res['tables'].items())})")
    for n in res["notes"]:
        log(f"note: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
