"""Download human-written Hindi text from a public Hugging Face dataset into texts/hi.txt.

Default: the Hindi side of the IIT Bombay English-Hindi corpus (cfilt/iitb-english-hindi),
test + validation splits: human-written/translated news, released 2018, so it predates
LLM-generated text. Any other dataset works with --dataset/--config/--splits/--field, e.g.

  # Hindi Wikipedia (Nov 2023 dump)
  python fetch_texts.py --dataset wikimedia/wikipedia --config 20231101.hi --splits train --field text

Text is kept byte-for-byte as the dataset stores it (no Unicode normalisation), apart from
turning newlines inside a record into spaces. The source is recorded in texts/hi.source.json.

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
    ap.add_argument("--splits", nargs="+", default=["test", "validation"])
    ap.add_argument("--field", default="translation.hi", help="dotted path to the Hindi text in a row")
    ap.add_argument("--n-lines", type=int, default=3000, help="records to take (drift.py groups them into passages)")
    ap.add_argument("--out", default=str(ROOT / "texts" / "hi.txt"))
    args = ap.parse_args()

    from datasets import load_dataset

    lines: list[str] = []
    for split in args.splits:
        ds = load_dataset(args.dataset, args.config, split=split, streaming=True)
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
    src = {"dataset": args.dataset, "config": args.config, "splits": args.splits,
           "field": args.field, "n_lines": len(lines)}
    out.with_suffix(".source.json").write_text(json.dumps(src, indent=2))
    print(f"wrote {len(lines)} lines from {args.dataset} -> {out}")


if __name__ == "__main__":
    main()
