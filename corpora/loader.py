"""One entry point over several corpora: iterate documents or batches, optionally shuffled."""
from __future__ import annotations

import random
from typing import Any, Callable, Iterable, Iterator

from . import registry
from .base import Corpus, Document


class CorpusLoader:
    """Iterate Documents from one or more registered corpora.

        loader = CorpusLoader(["samalochan", "hindi-aesthetics"], batch_size=8, shuffle=True, seed=0)
        for batch in loader:          # list[Document] of up to 8
            ...

        for doc in CorpusLoader("all"):   # batch_size=None yields single Documents
            ...

    datasets    a name, a list of names, "all", or Corpus objects (mix freely)
    batch_size  None for single documents, or an int for lists of documents
    shuffle     shuffle the combined index each epoch (seeded by `seed` + epoch)
    transform   optional Document -> anything, applied before batching
    filter      optional Document -> bool; documents for which it is False are skipped
    **corpus_kwargs  passed to every corpus built by name: normalize, dedupe, min_words
    """

    def __init__(self, datasets: str | Corpus | Iterable[str | Corpus] = "all", *, batch_size: int | None = None,
                 shuffle: bool = False, seed: int = 0, drop_last: bool = False,
                 transform: Callable[[Document], Any] | None = None,
                 filter: Callable[[Document], bool] | None = None, **corpus_kwargs: Any):
        if isinstance(datasets, (str, Corpus)):
            datasets = registry.available() if datasets == "all" else [datasets]
        self.corpora: list[Corpus] = [d if isinstance(d, Corpus) else registry.get(d, **corpus_kwargs)
                                      for d in datasets]
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size, self.shuffle, self.seed, self.drop_last = batch_size, shuffle, seed, drop_last
        self.transform, self.filter = transform, filter
        self.epoch = 0

    @property
    def index(self) -> list[tuple[int, int]]:
        """(corpus position, document position) for every document, in corpus order."""
        return [(ci, di) for ci, c in enumerate(self.corpora) for di in range(len(c))]

    def __len__(self) -> int:
        """Number of documents (or batches), before any `filter`."""
        n = len(self.index)
        if self.batch_size is None:
            return n
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)

    def documents(self) -> Iterator[Any]:
        order = self.index
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for ci, di in order:
            doc = self.corpora[ci][di]
            if self.filter and not self.filter(doc):
                continue
            yield self.transform(doc) if self.transform else doc

    def __iter__(self) -> Iterator[Any]:
        try:
            if self.batch_size is None:
                yield from self.documents()
                return
            batch: list[Any] = []
            for item in self.documents():
                batch.append(item)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if batch and not self.drop_last:
                yield batch
        finally:
            self.epoch += 1

    def __repr__(self) -> str:
        return f"CorpusLoader({[c.name for c in self.corpora]}, batch_size={self.batch_size}, shuffle={self.shuffle})"
