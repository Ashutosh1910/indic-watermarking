"""Small helpers shared by generate.py, trim.py and verify_generations.py.

Standard library only, apart from the import of the repo's ``synthid_wm`` library, which lives in
``../../synthid`` and is put on ``sys.path`` by :func:`add_synthid_to_path` (no ``pip install`` needed;
setting ``PYTHONPATH=<repo>/synthid`` works too).
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTHID_DIR = REPO_ROOT / "synthid"


def add_synthid_to_path() -> None:
    """Make ``import synthid_wm`` work from a plain checkout of the repo."""
    p = str(SYNTHID_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- JSONL


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Rows of a JSONL file; blank lines are skipped. A malformed line raises with its line number."""
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{i}: invalid JSON ({e})") from None


def read_jsonl_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(iter_jsonl(p))
    return rows


def dumps(row: dict[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(dumps(r) + "\n")
            n += 1
    return n


def read_id_list(path: str | Path) -> set[str]:
    """Ids from a text file (one per line, '#' comments allowed) or a JSONL file with an "id" field."""
    ids: set[str] = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            ids.add(str(json.loads(line)["id"]))
        else:
            ids.add(line)
    return ids


# --------------------------------------------------------------------------- row filters


def _as_filter_str(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def parse_filters(specs: Iterable[str] | None) -> dict[str, set[str]]:
    """``["language=hi", "level=G", "level=C_continue"]`` -> {"language": {"hi"}, "level": {"G", "C_continue"}}.

    Repeating a key means OR (any of the values); different keys are ANDed. A value may also be a
    comma-separated list (``level=G,C_continue``). Dotted keys reach into nested dicts
    (``generation.seed=1``). Values compare as strings; ``null``/``true``/``false`` match JSON null/bools.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for spec in specs or []:
        key, sep, value = spec.partition("=")
        if not sep or not key:
            raise ValueError(f"--filter expects key=value, got {spec!r}")
        out[key.strip()].update(v.strip() for v in value.split(","))
    return dict(out)


def get_dotted(row: dict[str, Any], key: str) -> Any:
    cur: Any = row
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def row_matches(row: dict[str, Any], filters: dict[str, set[str]]) -> bool:
    return all(_as_filter_str(get_dotted(row, k)) in vals for k, vals in filters.items())


# --------------------------------------------------------------------------- hashing / grouping


def stable_hash(*parts: Any, bits: int = 63) -> int:
    """Deterministic non-negative int from any JSON-serialisable parts (unlike Python's hash())."""
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & ((1 << bits) - 1)


def sibling(path: str | Path, suffix: str) -> Path:
    """``runs/en.jsonl`` + ``.config.yaml`` -> ``runs/en.config.yaml`` (drops a .jsonl/.json suffix)."""
    p = Path(path)
    stem = p.name[: -len(p.suffix)] if p.suffix in (".jsonl", ".json") else p.name
    return p.with_name(stem + suffix)


def md_table(header: list[str], rows: list[list[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return "" if v is None else str(v)

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)
