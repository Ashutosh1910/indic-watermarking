"""Side-by-side comparison of two drift runs (e.g. outputs/hi vs outputs/en).

Reads per_passage.csv and mismatches.csv from each run directory and writes one Markdown file with
the token fertility of each language, a hi/en table per metric and the top mismatches per language.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

LANG_NAMES = {"hi": "Hindi", "en": "English"}


def load(run: Path):
    rows = list(csv.DictReader((run / "per_passage.csv").open(encoding="utf-8")))
    for r in rows:
        for k, v in r.items():
            if k != "perturbation":
                r[k] = float(v)
    mm = list(csv.DictReader((run / "mismatches.csv").open(encoding="utf-8")))
    return rows, mm


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return statistics.fmean(xs) if xs else float("nan")


def f(x, nd=3):
    return "–" if math.isnan(x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="run dir for language A (e.g. outputs/hi)")
    ap.add_argument("--b", required=True, help="run dir for language B (e.g. outputs/en)")
    ap.add_argument("--la", default=None, help="label for A (default: dir name)")
    ap.add_argument("--lb", default=None, help="label for B (default: dir name)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-examples", type=int, default=5)
    args = ap.parse_args()
    a, b = Path(args.a), Path(args.b)
    la = args.la or a.name.removeprefix("quick-")
    lb = args.lb or b.name.removeprefix("quick-")
    runs = {la: load(a), lb: load(b)}

    by = {lang: defaultdict(list) for lang in runs}
    for lang, (rows, _) in runs.items():
        for r in rows:
            by[lang][r["perturbation"]].append(r)
    perts = list(dict.fromkeys(r["perturbation"] for r in runs[la][0]))

    md = [f"# Retokenization drift: {LANG_NAMES.get(la, la)} vs {LANG_NAMES.get(lb, lb)}", ""]
    md += ["| | " + " | ".join(LANG_NAMES.get(l, l) for l in runs) + " |", "|---|---|---|"]
    stats = {}
    for lang in runs:
        g = by[lang]["none"] if "none" in by[lang] else by[lang][perts[0]]
        toks, chars, words = (sum(r[k] for r in g) for k in ("len_orig", "chars_orig", "words_orig"))
        stats[lang] = (len(g), toks / len(g), chars / toks, toks / words)
    md.append("| passages | " + " | ".join(str(stats[l][0]) for l in runs) + " |")
    md.append("| tokens / passage | " + " | ".join(f"{stats[l][1]:.0f}" for l in runs) + " |")
    md.append("| chars / token | " + " | ".join(f"{stats[l][2]:.2f}" for l in runs) + " |")
    md.append("| tokens / word | " + " | ".join(f"{stats[l][3]:.2f}" for l in runs) + " |")
    for lang, d in ((la, a), (lb, b)):
        sm = d / "summary.md"
        if sm.exists():
            src = next((l for l in sm.read_text(encoding="utf-8").splitlines() if l.startswith("Text source")), "")
            md.append(f"\n{LANG_NAMES.get(lang, lang)}: {src}")
    if stats[la][0] != stats[lb][0]:
        md.append(f"\n**Warning:** unequal passage counts ({stats[la][0]} vs {stats[lb][0]}).")

    md += ["", "## All passages", "",
           f"Each cell is {la} / {lb}.", "",
           "| perturbation | text changed | exact match | aligned pos. | 5-gram survival | Δ tokens |",
           "|---|---|---|---|---|---|"]
    for p in perts:
        cells = []
        for key, nd in (("text_changed", 3), ("exact", 3), ("aligned_frac", 3), ("ngram5_survival", 3)):
            cells.append(" / ".join(f(mean([r[key] for r in by[l][p]]), nd) for l in runs))
        cells.append(" / ".join(f(mean([r["len_new"] - r["len_orig"] for r in by[l][p]]), 2) for l in runs))
        md.append(f"| {p} | " + " | ".join(cells) + " |")

    md += ["", "## Only passages the perturbation changed (cost per hit)", "",
           f"Each cell is {la} / {lb}; `–` means the perturbation never changed a passage in that language.", "",
           "| perturbation | n changed | chars changed / passage | exact match | aligned pos. | 5-gram survival |",
           "|---|---|---|---|---|---|"]
    for p in perts:
        ch = {l: [r for r in by[l][p] if r["text_changed"]] for l in runs}
        cells = [" / ".join(str(len(ch[l])) for l in runs)]
        cells.append(" / ".join(f(mean([r["chars_changed"] for r in ch[l]]), 1) for l in runs))
        for key in ("exact", "aligned_frac", "ngram5_survival"):
            cells.append(" / ".join(f(mean([r[key] for r in ch[l]])) for l in runs))
        md.append(f"| {p} | " + " | ".join(cells) + " |")

    esc = lambda s: s.replace("|", "\\|").replace("\n", "⏎")
    md += ["", "## Top mismatches", "",
           "Original span (code points), original tokens → re-encoded tokens, how often, and context."]
    for lang, (_, mm) in runs.items():
        md += ["", f"### {LANG_NAMES.get(lang, lang)}"]
        groups = defaultdict(list)
        for m in mm:
            groups[m["perturbation"]].append(m)
        if not groups:
            md += ["", "No mismatches for any perturbation."]
        for p in perts:
            if p not in groups:
                continue
            md += ["", f"**{p}**", "", "| count | text | code points | original tokens | re-encoded tokens | context |",
                   "|---|---|---|---|---|---|"]
            for m in groups[p][: args.n_examples]:
                md.append(f"| {m['count']} | `{esc(m['text'])}` | {m['codepoints'][:60]} | `{esc(m['orig_tokens'])}` | "
                          f"`{esc(m['new_tokens'])}` | {esc(m['context'])} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
