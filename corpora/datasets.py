"""The registered corpora. To add one, see the README in this folder: usually a single register_text_dir call."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import TextDirCorpus
from .registry import register


@register("hindi-aesthetics", root="Hindi-Aesthetics-Corpus",
          description="Hindi novels, stories and essays (hindisamay, premchand.co.in, borilib) via "
                      "gayatrivenugopal/Hindi-Aesthetics-Corpus. Punctuation and Latin removed upstream; "
                      "one sentence per line.")
class HindiAesthetics(TextDirCorpus):
    # Files come both with and without a .txt extension, so take every file.
    pattern = "*"

    def _title(self, path: Path, text: str) -> str:
        stem = path.stem if path.suffix == ".txt" else path.name
        return stem.replace("---", " - ").replace("-", " ").strip()

    def _meta(self, path: Path, text: str) -> dict[str, Any]:
        return {
            "source": "Hindi-Aesthetics-Corpus",
            "has_txt_extension": path.suffix == ".txt",
            # Upstream preprocessing: no punctuation, sentences on separate lines, removed punctuation
            # left double spaces. Useful to know before running whitespace-sensitive experiments.
            "punctuation_removed": True,
        }


@register("samalochan", root="Samalochan",
          description="Posts from samalochan.com published before 2022-01-01, more than 500 words, "
                      "punctuation kept. File name: <publish-date>_<post-slug>.txt.")
class Samalochan(TextDirCorpus):
    pattern = "*.txt"
    filename_regex = r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<slug>.+)\.txt$"

    def _title(self, path: Path, text: str) -> str:
        # The slug is the post title cut short. The first line is not reliable: it is sometimes a
        # section label such as "उर्दू से :", so it goes to meta["first_line"] instead.
        slug = super()._meta(path, text).get("slug", path.stem)
        return slug.replace("-", " ").strip()

    def _meta(self, path: Path, text: str) -> dict[str, Any]:
        meta = super()._meta(path, text)
        meta["first_line"] = next((line.strip() for line in text.splitlines() if line.strip()), "")
        if "date" in meta:
            meta["year"] = int(meta["date"][:4])
        meta["source"] = "samalochan.com"
        meta["punctuation_removed"] = False
        return meta
