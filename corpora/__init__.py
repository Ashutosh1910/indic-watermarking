"""Unified access to the text corpora under data/.

    import corpora
    corpora.available()        # ['hindi-aesthetics', 'hindisamay', 'medium', 'samalochan']
    sam = corpora.get("samalochan")                      # a Sequence[Document], raw text
    doc = sam[0]; doc.text, doc.title, doc.meta["date"]
    for batch in corpora.CorpusLoader("all", batch_size=16, shuffle=True):
        ...

Nothing under data/ is modified; text is returned byte-for-byte unless `normalize=` is given.
"""
from . import datasets as _datasets  # noqa: F401  (registers the built-in corpora)
from .base import DATA_ROOT, Corpus, Document, TextDirCorpus
from .loader import CorpusLoader
from .registry import available, describe, get, register, register_text_dir

__all__ = ["Corpus", "CorpusLoader", "DATA_ROOT", "Document", "TextDirCorpus", "available", "describe",
           "get", "register", "register_text_dir"]
