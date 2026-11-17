"""Retokenization drift on human text (Hindi, English, ...), tokenizer only (no model).

For every passage x (a few sentences, truncated to --max-tokens at a character boundary):

    ids  = encode(x)                       # the tokenizer's canonical split of human text
    text = decode(ids)
    ids2 = encode(perturb(text))           # what a detector would see after copy-paste etc.

and we compare ids with ids2:

  exact            ids2 == ids
  aligned_frac     share of positions in ids inside difflib matching blocks
  ngram5_survival  share of ids' 5-token windows (multiset) still present in ids2. A SynthID
                   g-value at position t depends only on tokens t-4..t, so this is the share of
                   scorable positions that would keep their watermark signal.

Perturbations model what happens to text between a chat window and a detector.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from common import DEFAULT_TOKENIZER, NGRAM_LEN, ROOT
from transformers import AutoTokenizer

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"))  # ZWSP, ZWNJ, ZWJ, WJ, BOM
SPACES = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")          # NBSP and other spaces

# Devanagari letters with nukta. Unicode keeps the precomposed forms (U+0958-095F) out of NFC,
# so NFC/NFD *decompose* them; many keyboards and older web text produce the precomposed form.
NUKTA_PRECOMPOSED = {unicodedata.normalize("NFD", chr(c)): chr(c) for c in range(0x0958, 0x0960)}


def compose_nukta(s: str) -> str:
    for dec, pre in NUKTA_PRECOMPOSED.items():
        s = s.replace(dec, pre)
    return s


def whitespace(s: str) -> str:
    return re.sub(r" {2,}", " ", SPACES.sub(" ", s)).strip()


PERTURBATIONS = {
    "none": lambda s: s,                                         # pure decode -> encode
    "NFC": lambda s: unicodedata.normalize("NFC", s),
    "NFD": lambda s: unicodedata.normalize("NFD", s),
    "NFKC": lambda s: unicodedata.normalize("NFKC", s),
    "strip_zw": lambda s: s.translate(ZERO_WIDTH),               # ZWJ/ZWNJ dropped (many sanitisers)
    "whitespace": whitespace,                                    # NBSP -> space, collapse runs, trim
    "nukta_precomposed": compose_nukta,                          # क़ as one code point
    "webform": lambda s: whitespace(unicodedata.normalize("NFKC", s).translate(ZERO_WIDTH)),
}


LANG_NAMES = {"hi": "Hindi", "en": "English"}


def ngram_survival(a: list[int], b: list[int], n: int = NGRAM_LEN) -> float:
    wa = Counter(tuple(a[i : i + n]) for i in range(len(a) - n + 1))
    wb = Counter(tuple(b[i : i + n]) for i in range(len(b) - n + 1))
    total = sum(wa.values())
    return sum((wa & wb).values()) / total if total else float("nan")


def aligned(a: list[int], b: list[int]):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(m.size for m in sm.get_matching_blocks()) / len(a), sm.get_opcodes()


def token_strings(tok, ids: list[int]) -> str:
    out = []
    for i in ids:
        s = tok.decode([i], clean_up_tokenization_spaces=False)
        out.append(f"{s}#{i}" if "\ufffd" in s else s.replace(" ", "␣"))
    return " · ".join(out)


def codepoints(s: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in s)


def make_passages(lines: list[str], tok, max_tokens: int, n: int) -> list[str]:
    """Join consecutive lines until a passage reaches max_tokens, then trim it at a character
    boundary (never inside a character) so it encodes to <= max_tokens."""
    out, buf = [], ""
    for line in lines:
        buf = f"{buf} {line}".strip() if buf else line
        if len(tok.encode(buf, add_special_tokens=False)) >= max_tokens:
            while len(tok.encode(buf, add_special_tokens=False)) > max_tokens:
                buf = buf[: int(len(buf) * 0.95)].rstrip()  # no trailing space from the cut
            out.append(buf)
            buf = ""
            if len(out) >= n:
                break
    if buf and len(out) < n:
        out.append(buf)
    return out


def mean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return statistics.fmean(xs) if xs else float("nan")


def fmt(x, nd=3):
    return "–" if isinstance(x, float) and math.isnan(x) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--texts", default=str(ROOT / "texts" / "hi.txt"), help="UTF-8 file, one line per sentence/paragraph")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--n-passages", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS), choices=list(PERTURBATIONS))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "hi"))
    ap.add_argument("--n-examples", type=int, default=20)
    ap.add_argument("--quick", action="store_true", help="smoke test: 10 passages")
    ap.add_argument("--lang", default=None, help="label for the report (default: the texts file name, e.g. hi)")
    args = ap.parse_args()
    args.lang = args.lang or Path(args.texts).stem
    lang_name = LANG_NAMES.get(args.lang, args.lang)
    if args.quick:
        args.n_passages = 10

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lines = [l.strip() for l in Path(args.texts).read_text(encoding="utf-8").splitlines() if l.strip()]
    passages = make_passages(lines, tok, args.max_tokens, args.n_passages)
    backend = getattr(tok, "backend_tokenizer", None)
    normalizer = repr(backend.normalizer) if backend is not None and backend.normalizer is not None else "none"
    if len(passages) < args.n_passages:
        print(f"WARNING: only {len(passages)} passages of {args.max_tokens} tokens available "
              f"(asked for {args.n_passages}); fetch more lines")
    print(f"{len(passages)} passages from {args.texts}, tokenizer={args.tokenizer}, built-in normalizer={normalizer}")

    rows = []
    spans = defaultdict(Counter)   # perturbation -> Counter[(orig text, orig toks, new toks)]
    context = {}
    for pi, x in enumerate(passages):
        a = tok.encode(x, add_special_tokens=False)
        text = tok.decode(a, clean_up_tokenization_spaces=False)
        for name in args.perturbations:
            t2 = PERTURBATIONS[name](text)
            b = tok.encode(t2, add_special_tokens=False)
            al, ops = aligned(a, b)
            changed = t2 != text
            rows.append({
                "passage": pi, "perturbation": name, "len_orig": len(a), "len_new": len(b),
                "chars_orig": len(text), "words_orig": len(text.split()),
                "text_changed": int(changed), "chars_changed": sum(1 for c in difflib.ndiff(text, t2) if c[0] in "+-")
                if changed and len(text) < 5000 else 0,
                "exact": int(a == b), "aligned_frac": al, "ngram5_survival": ngram_survival(a, b),
            })
            for tag, i1, i2, j1, j2 in ops:
                if tag == "equal":
                    continue
                key = (tok.decode(a[i1:i2]), token_strings(tok, a[i1:i2]), token_strings(tok, b[j1:j2]))
                spans[name][key] += 1
                context.setdefault((name, key), tok.decode(a[max(0, i1 - 4) : i2 + 4]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src_file = Path(args.texts).with_suffix(".source.json")
    source = src_file.read_text().strip() if src_file.exists() else f"file {args.texts}"
    with (out / "per_passage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(list)
    for r in rows:
        by[r["perturbation"]].append(r)
    lines_md = [
        f"# Retokenization drift, {lang_name} human text", "",
        f"Tokenizer `{args.tokenizer}` · {len(passages)} passages from `{Path(args.texts).name}` · "
        f"mean {mean([r['len_orig'] for r in by[args.perturbations[0]]]):.0f} tokens/passage · n-gram window {NGRAM_LEN}", "",
        f"Text source: `{' '.join(source.split())}`", "",
        f"Tokenizer's built-in normalizer: `{normalizer}` (if this is NFC, NFC/NFD/nukta changes are undone "
        "inside encode() and should show no drift).", "",
        "**All passages** (what you'd see on a random document)", "",
        "| perturbation | text changed | exact match | aligned pos. | 5-gram survival | Δ tokens |",
        "|---|---|---|---|---|---|",
    ]
    for name in args.perturbations:
        g = by[name]
        lines_md.append(f"| {name} | {fmt(mean([r['text_changed'] for r in g]))} | {fmt(mean([r['exact'] for r in g]))} | "
                        f"{fmt(mean([r['aligned_frac'] for r in g]))} | {fmt(mean([r['ngram5_survival'] for r in g]))} | "
                        f"{fmt(mean([r['len_new'] - r['len_orig'] for r in g]), 2)} |")
    lines_md += ["", "**Only passages the perturbation actually changed** (cost per hit)", "",
                 "| perturbation | n changed | chars changed / passage | exact match | aligned pos. | 5-gram survival |",
                 "|---|---|---|---|---|---|"]
    for name in args.perturbations:
        g = [r for r in by[name] if r["text_changed"]]
        lines_md.append(f"| {name} | {len(g)} | {fmt(mean([r['chars_changed'] for r in g]), 1)} | "
                        f"{fmt(mean([r['exact'] for r in g]))} | {fmt(mean([r['aligned_frac'] for r in g]))} | "
                        f"{fmt(mean([r['ngram5_survival'] for r in g]))} |")
    (out / "summary.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    print("\n".join(lines_md))

    esc = lambda s: s.replace("|", "\\|").replace("\n", "⏎")
    ml = ["# Most frequent token mismatches", "",
          "Original span text (with code points), original tokens → re-encoded tokens, count, context.", ""]
    for name in args.perturbations:
        c = spans.get(name)
        if not c:
            continue
        ml += [f"## {name} ({sum(c.values())} mismatched spans)", "",
               "| count | text | code points | original tokens | re-encoded tokens | context |",
               "|---|---|---|---|---|---|"]
        for key, cnt in c.most_common(args.n_examples):
            txt, o, n_ = key
            ml.append(f"| {cnt} | `{esc(txt)}` | {codepoints(txt)[:80]} | `{esc(o)}` | `{esc(n_)}` | {esc(context[(name, key)])} |")
        ml.append("")
    (out / "mismatches.md").write_text("\n".join(ml) + "\n", encoding="utf-8")
    with (out / "mismatches.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "count", "text", "codepoints", "orig_tokens", "new_tokens", "context"])
        for name in args.perturbations:
            for key, cnt in spans.get(name, Counter()).most_common(args.n_examples):
                w.writerow([name, cnt, key[0], codepoints(key[0]), key[1], key[2], context[(name, key)]])
    print(f"wrote {out/'summary.md'}, {out/'mismatches.md'}, {out/'per_passage.csv'}")


if __name__ == "__main__":
    main()
