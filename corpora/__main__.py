"""Command line: python -m corpora list | stats [names...] | show <name> [index]"""
from __future__ import annotations

import argparse
import statistics

from . import available, describe, get


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m corpora", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="registered datasets")
    s = sub.add_parser("stats", help="documents, words and duplicates per dataset")
    s.add_argument("names", nargs="*")
    sh = sub.add_parser("show", help="print one document")
    sh.add_argument("name")
    sh.add_argument("index", type=int, nargs="?", default=0)
    sh.add_argument("--chars", type=int, default=600)
    args = ap.parse_args()

    if args.cmd == "list":
        for name, desc in describe().items():
            print(f"{name:20s} {desc}")
    elif args.cmd == "stats":
        for name in args.names or available():
            c = get(name)
            words = [d.n_words for d in c]
            print(f"{name}: {len(c)} documents, {sum(words):,} words, median {statistics.median(words):,.0f} "
                  f"words/doc, {len(c.duplicates)} exact duplicates dropped")
            for dup, orig in c.duplicates.items():
                print(f"    {dup.name} == {orig.name}")
    else:
        d = get(args.name)[args.index]
        print(f"id: {d.id}\ntitle: {d.title}\nmeta: {d.meta}\nwords: {d.n_words}\n---\n{d.text[:args.chars]}")


if __name__ == "__main__":
    main()
