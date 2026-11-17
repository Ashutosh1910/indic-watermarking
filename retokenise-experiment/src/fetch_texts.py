"""Download human-written text (Hindi or English) from a public Hugging Face dataset into texts/<lang>.txt.

Default: the Hindi (--lang hi) or English (--lang en) side of the IIT Bombay English-Hindi corpus (cfilt/iitb-english-hindi),
test + validation splits: human-written/translated news, released 2018, so it predates
LLM-generated text. Any other dataset works with --dataset/--config/--splits/--field, e.g.

  # Hindi Wikipedia (Nov 2023 dump)
  python fetch_texts.py --dataset wikimedia/wikipedia --config 20231101.hi --splits train --field text

  # Web-crawled Hindi: mC4 (Common Crawl, up to 2020) and AI4Bharat Sangraha (verified web split)
  python fetch_texts.py --lang hi --dataset allenai/c4 --config hi --splits train --field text \
      --n-lines 1500 --out ../texts/hi-mc4.txt
  python fetch_texts.py --lang hi --dataset ai4bharat/sangraha --data-dir verified/hin --splits train \
      --field text --n-lines 1500 --out ../texts/hi-sangraha.txt

Text is kept byte-for-byte as the dataset stores it (no Unicode normalisation), apart from
turning newlines inside a record into spaces. The source is recorded in texts/<lang>.source.json.

Because IITB is a parallel corpus, the English and Hindi arms share the same news content, period
and pre-LLM provenance; one side of each pair is a human translation of the other.

To use a corpus you downloaded yourself (e.g. IndicCorp's hi.txt), skip this script:
  head -n 3000 /path/to/hi.txt > texts/hi.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import ROOT


def get_field(row: dict, dotted: str):
    for part in dotted.split("."):
        row = row[part]
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="cfilt/iitb-english-hindi")
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None, help="dataset sub-folder, e.g. verified/hin for ai4bharat/sangraha")
    ap.add_argument("--splits", nargs="+", default=["test", "validation"])
    ap.add_argument("--lang", default="hi", help="language code; sets the default --field and --out")
    ap.add_argument("--field", default=None, help="dotted path to the text in a row (default translation.<lang>)")
    ap.add_argument("--n-lines", type=int, default=3000, help="records to take (drift.py groups them into passages)")
    ap.add_argument("--out", default=None, help="default texts/<lang>.txt")
    args = ap.parse_args()
    args.field = args.field or f"translation.{args.lang}"
    args.out = args.out or str(ROOT / "texts" / f"{args.lang}.txt")

    from datasets import load_dataset

    lines: list[str] = []
    for split in args.splits:
        ds = load_dataset(args.dataset, args.config, split=split, streaming=True,
                          **({"data_dir": args.data_dir} if args.data_dir else {}))
        for row in ds:
            t = get_field(row, args.field)
            if t and t.strip():
                lines.append(t.replace("\r", " ").replace("\n", " "))
            if len(lines) >= args.n_lines:
                break
        if len(lines) >= args.n_lines:
            break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    src = {"lang": args.lang, "dataset": args.dataset, "config": args.config, "data_dir": args.data_dir, "splits": args.splits,
           "field": args.field, "n_lines": len(lines)}
    out.with_suffix(".source.json").write_text(json.dumps(src, indent=2))
    print(f"wrote {len(lines)} lines from {args.dataset} -> {out}")


if __name__ == "__main__":
    main()
