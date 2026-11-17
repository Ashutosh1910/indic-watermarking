"""Dataset registry: one name per corpus, so callers never need to know paths or file layouts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, TypeVar

from .base import DATA_ROOT, Corpus, TextDirCorpus

_REGISTRY: dict[str, type[Corpus]] = {}
C = TypeVar("C", bound=type[Corpus])


def register(name: str, *, root: str | Path | None = None, description: str = "") -> Callable[[C], C]:
    """Class decorator. `root` is relative to data/ unless absolute.

        @register("my-corpus", root="My-Corpus", description="...")
        class MyCorpus(TextDirCorpus):
            pattern = "*.txt"
    """
    def deco(cls: C) -> C:
        if name in _REGISTRY:
            raise ValueError(f"dataset {name!r} is already registered")
        cls.name = name
        if root is not None:
            cls.default_root = Path(root) if Path(root).is_absolute() else DATA_ROOT / root
        if description:
            cls.description = description
        _REGISTRY[name] = cls
        return cls
    return deco


def register_text_dir(name: str, root: str | Path, *, description: str = "", pattern: str = "*",
                      filename_regex: str | None = None, exclude: set[str] | frozenset[str] = frozenset()
                      ) -> type[Corpus]:
    """Register a plain folder of text files without writing a class."""
    cls = type(f"TextDir_{name}", (TextDirCorpus,),
               {"pattern": pattern, "filename_regex": filename_regex, "exclude": frozenset(exclude)})
    return register(name, root=root, description=description)(cls)


def available() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str, **kwargs: Any) -> Corpus:
    """Instantiate a registered corpus. kwargs go to the Corpus constructor (root, normalize, dedupe, min_words)."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(available())}") from None
    return cls(**kwargs)


def describe() -> dict[str, str]:
    return {n: _REGISTRY[n].description for n in available()}
