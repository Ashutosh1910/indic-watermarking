"""Collect human-written documents into one JSONL for false-positive checks (and as build_edit_prompts input).

Each ``--dir`` is a folder of text files (one document per file), given as ``NAME:LANG:PATH``, e.g.

    python promptset/scripts/build_human_pool.py --out work/human_pool.jsonl \\
        --dir hindisamay:hi:data/HindiSamay --dir samalochan:hi:data/Samalochan --dir medium:en:data/Medium

Optional per-folder suffix ``:linebreak`` marks text whose punctuation was stripped upstream (one sentence
per line, e.g. ``hindi-aesthetics:hi:data/Hindi-Aesthetics-Corpus:linebreak``); its rows get
``sentence_end_rule: punct+linebreak`` so trim.py treats every line break as a sentence end.

What is done to each file (nothing else; the text is never rewritten):

* ``README*``, ``*.md``, ``*.py``, ``*.csv`` files and hidden files are skipped.
* The heading lines (title, author, "____" separators) are dropped with build_edit_prompts.strip_header,
  the same rule the edit-level builder uses (``meta.header_lines_stripped``). ``--keep-header`` turns it off.
* ``genre`` and ``title`` come from the folder's ``metadata.csv`` (column ``file``), when there is one.
* Documents under ``--min-words`` (default 500) are skipped; exact duplicate texts are skipped.

Row: ``id`` (``<name>/<file name>``, unique), ``pair_id`` (same), ``language``, ``level: "human"``, ``source``
(= name), ``source_id`` (file name), ``genre``, ``title``, ``text``, ``n_words``, ``sentence_end_rule``,
``meta``. The same fields are what ``build_edit_prompts.py build --jsonl`` reads.

The output holds the corpus texts: keep it out of git (the default location ``work/`` is gitignored).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import log, write_jsonl  # noqa: E402
from build_edit_prompts import strip_header  # noqa: E402

SKIP_SUFFIXES = {".md", ".py", ".csv", ".json", ".jsonl", ".yaml", ".yml"}


def parse_dir(spec: str) -> tuple[str, str, Path, str]:
    parts = spec.split(":")
    if len(parts) < 3:
        raise SystemExit(f"--dir expects NAME:LANG:PATH[:linebreak], got {spec!r}")
    rule = "punct"
    if parts[-1] == "linebreak":
        rule = "punct+linebreak"
        parts = parts[:-1]
    name, lang, path = parts[0], parts[1], ":".join(parts[2:])
    return name, lang, Path(path), rule


def read_metadata(folder: Path) -> dict[str, dict[str, str]]:
    p = folder / "metadata.csv"
    if not p.exists():
        return {}
    with open(p, encoding="utf-8-sig", newline="") as f:
        return {row["file"]: row for row in csv.DictReader(f) if row.get("file")}


def collect(name: str, lang: str, folder: Path, rule: str, min_words: int, keep_header: bool,
            seen: set[str]) -> tuple[list[dict[str, Any]], Counter]:
    if not folder.is_dir():
        raise SystemExit(f"{name}: folder not found: {folder}")
    meta_rows = read_metadata(folder)
    rows: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith(".") or path.name.upper().startswith("README") \
                or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        raw = path.read_text(encoding="utf-8", errors="strict")
        text, n_hdr, hdr = (raw, 0, "") if keep_header else strip_header(raw)
        text = text.strip()
        n_words = len(text.split())
        if n_words < min_words:
            skipped[f"< {min_words} words"] += 1
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            skipped["duplicate text"] += 1
            continue
        seen.add(digest)
        m = meta_rows.get(path.name, {})
        title = m.get("title") or (hdr.splitlines()[0].strip() if hdr else path.stem)
        rows.append({
            "id": f"{name}/{path.name}", "pair_id": f"{name}/{path.name}", "language": lang, "level": "human",
            "source": name, "source_id": path.name, "genre": m.get("genre") or "unknown", "title": title,
            "n_words": n_words, "sentence_end_rule": rule,
            "meta": {"header_lines_stripped": n_hdr, "sha256": digest, "path": str(path),
                     **{k: v for k, v in m.items() if k not in ("file", "title", "genre")}},
            "text": text,
        })
    return rows, skipped


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True, metavar="NAME:LANG:PATH[:linebreak]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-words", type=int, default=500)
    ap.add_argument("--keep-header", action="store_true", help="do not strip title/author lines")
    args = ap.parse_args(argv)

    seen: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    for spec in args.dir:
        name, lang, folder, rule = parse_dir(spec)
        rows, skipped = collect(name, lang, folder, rule, args.min_words, args.keep_header, seen)
        genres = Counter(r["genre"] for r in rows)
        log(f"{name} ({lang}, {folder}): {len(rows)} docs; skipped {dict(skipped)}; genres {dict(genres.most_common(8))}")
        all_rows.extend(rows)
    write_jsonl(args.out, all_rows)
    log(f"wrote {args.out} ({len(all_rows)} docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
