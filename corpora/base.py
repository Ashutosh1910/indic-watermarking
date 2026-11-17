"""Core types: Document, the Corpus base class, and the text-directory corpus most datasets use."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"

Normalizer = Callable[[str], str]


@dataclass(frozen=True)
class Document:
    """One text from one corpus. `text` is exactly what the file holds unless a normalizer was asked for."""

    id: str                 # "<dataset>/<file name>", unique across all corpora
    dataset: str
    text: str
    title: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def resolve_normalizer(normalize: str | Normalizer | None) -> Normalizer | None:
    """None keeps text byte-for-byte; "NFC"/"NFD"/"NFKC"/"NFKD" or any str -> str callable is applied on read."""
    if normalize is None or callable(normalize):
        return normalize
    form = normalize.upper()
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"unknown normalization {normalize!r}")
    return lambda s: unicodedata.normalize(form, s)


class Corpus(Sequence[Document]):
    """A read-only, indexable collection of Documents.

    Subclasses implement `_paths()` (which files belong to the corpus) and may override `_meta()` and
    `_title()` for dataset-specific metadata. Indexing and len() work, so a Corpus can be passed straight to
    torch.utils.data.DataLoader as a map-style dataset; nothing here imports torch.

    Options shared by every corpus:
      normalize   None (default, raw text), a Unicode form name, or a callable applied to each text
      dedupe      drop files whose bytes exactly match an earlier file (sorted by path); default True
      min_words   skip documents with fewer whitespace-separated words
    """

    name: str = ""            # registry key, set by @register
    default_root: Path | None = None
    description: str = ""

    def __init__(self, root: str | Path | None = None, *, normalize: str | Normalizer | None = None,
                 dedupe: bool = True, min_words: int = 0):
        self.root = Path(root) if root is not None else self.default_root
        if self.root is None or not self.root.is_dir():
            raise FileNotFoundError(f"{self.name}: data folder not found: {self.root}")
        self.normalize = resolve_normalizer(normalize)
        self.dedupe = dedupe
        self.min_words = min_words
        self._index: list[Path] | None = None
        self.duplicates: dict[Path, Path] = {}   # dropped file -> the file it duplicates

    # ---- to override -------------------------------------------------------------------------------
    def _paths(self) -> list[Path]:
        raise NotImplementedError

    def _title(self, path: Path, text: str) -> str:
        return path.stem if path.suffix == ".txt" else path.name

    def _meta(self, path: Path, text: str) -> dict[str, Any]:
        return {}

    # ---- index -------------------------------------------------------------------------------------
    @property
    def paths(self) -> list[Path]:
        if self._index is None:
            paths = sorted(self._paths())
            if self.dedupe:
                seen: dict[str, Path] = {}
                kept = []
                for p in paths:
                    h = hashlib.sha1(p.read_bytes()).hexdigest()
                    if h in seen:
                        self.duplicates[p] = seen[h]
                    else:
                        seen[h] = p
                        kept.append(p)
                paths = kept
            if self.min_words:
                paths = [p for p in paths if len(self._read(p).split()) >= self.min_words]
            self._index = paths
        return self._index

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def load(self, path: Path) -> Document:
        raw = self._read(path)
        text = self.normalize(raw) if self.normalize else raw
        meta = {"filename": path.name, **self._meta(path, raw)}
        return Document(id=f"{self.name}/{path.name}", dataset=self.name, text=text,
                        title=self._title(path, raw), path=path, meta=meta)

    # ---- Sequence protocol ------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self.load(p) for p in self.paths[i]]
        return self.load(self.paths[i])

    def __iter__(self) -> Iterator[Document]:
        for p in self.paths:
            yield self.load(p)

    def filter(self, pred: Callable[[Document], bool]) -> Iterator[Document]:
        return (d for d in self if pred(d))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, root={str(self.root)!r})"


class TextDirCorpus(Corpus):
    """A folder of UTF-8 text files, one document per file.

    Class attributes (or the same keyword arguments to `register_text_dir`) configure it:
      pattern          glob relative to the root, e.g. "*.txt" or "**/*"; default "*" (every file, any extension)
      filename_regex   optional regex with named groups; the groups become metadata (e.g. a date)
      exclude          file names to skip (README.md and dotfiles are always skipped)
    """

    pattern: str = "*"
    filename_regex: str | None = None
    exclude: frozenset[str] = frozenset()

    def _paths(self) -> list[Path]:
        skip = {"README", "README.md", "LICENSE", *self.exclude}
        return [p for p in self.root.glob(self.pattern)
                if p.is_file() and not p.name.startswith(".") and p.name not in skip]

    def _meta(self, path: Path, text: str) -> dict[str, Any]:
        if self.filename_regex and (m := re.match(self.filename_regex, path.name)):
            return {k: v for k, v in m.groupdict().items() if v is not None}
        return {}
