"""The registered corpora. To add one, see the README in this folder: usually a single register_text_dir call."""
from __future__ import annotations

import csv
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
    # samalochan.com has no genre field. meta["genre"] follows a fixed rule (see samalochan_genre).
    STORY_SLUG_PREFIX = "कथा-गाथा"                 # the site's fiction column ("katha-gatha")
    STORY_LABELS = frozenset({"कहानी", "कहानियाँ", "कहानियां", "लम्बी कहानी", "लंबी कहानी"})

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
        meta["genre"], meta["genre_rule"] = samalochan_genre(meta.get("slug", path.stem), text)
        return meta


def samalochan_genre(slug: str, text: str) -> tuple[str, str]:
    """("story" | "essay", the rule that decided). samalochan.com has no genre metadata, so this is a fixed,
    deterministic rule on the file name and the first lines, nothing else:

      story  the slug starts with "कथा-गाथा" (the site's fiction column), or one of the first three non-empty
             lines is exactly a section label "कहानी" / "कहानियाँ" / "लम्बी कहानी" (the site's label for a story)
      essay  everything else: criticism, essays, memoirs, interviews, reviews. This is a residual class: a story
             posted without that column name or label (an editor's note introducing a new story) stays "essay".
    """
    if slug.startswith(Samalochan.STORY_SLUG_PREFIX):
        return "story", "slug prefix कथा-गाथा"
    first = [ln.strip() for ln in text.splitlines() if ln.strip()][:3]
    if any(ln in Samalochan.STORY_LABELS for ln in first):
        return "story", "section label"
    return "essay", "default (no story marker)"


class TitleAuthorCorpus(TextDirCorpus):
    """A folder of "<title>\\n\\n<author>\\n\\n<paragraphs...>" text files plus a metadata.csv with one row per
    file (column "file"). The title and author come from metadata.csv (falling back to the first and second
    lines); `meta["genre"]` is the coarse study genre ("story" | "essay") and `meta["genre_fine"]` the source's
    own label. Subclasses set `author_field` and implement `_genre`."""

    pattern = "*.txt"
    metadata_file = "metadata.csv"
    author_field = "author"
    source_name = ""
    int_fields: tuple[str, ...] = ("words",)

    def _metadata(self) -> dict[str, dict[str, str]]:
        if getattr(self, "_meta_rows", None) is None:
            p = self.root / self.metadata_file
            rows: dict[str, dict[str, str]] = {}
            if p.is_file():
                with open(p, encoding="utf-8-sig", newline="") as f:      # HindiSamay's file starts with a BOM
                    rows = {r["file"]: r for r in csv.DictReader(f)}
            self._meta_rows = rows
        return self._meta_rows

    @staticmethod
    def _header(text: str) -> list[str]:
        return [ln.strip() for ln in text.split("\n")[:3:2]]          # lines 1 and 3: title, author

    def _title(self, path: Path, text: str) -> str:
        row = self._metadata().get(path.name) or {}
        return (row.get("title") or self._header(text)[0]).strip()

    def _genre(self, fine: str) -> str:
        raise NotImplementedError

    def _meta(self, path: Path, text: str) -> dict[str, Any]:
        meta = super()._meta(path, text)
        row = dict(self._metadata().get(path.name) or {})
        row.pop("file", None)
        hdr = self._header(text)
        meta.update(row)
        for k in self.int_fields:
            if str(meta.get(k, "")).isdigit():
                meta[k] = int(meta[k])
        meta["author"] = (row.get(self.author_field) or (hdr[1] if len(hdr) > 1 else "")).strip()
        meta["first_line"] = hdr[0] if hdr else ""
        fine = (row.get("genre") or "").strip()
        meta["genre_fine"] = fine or "unknown"
        meta["genre"] = self._genre(fine) if fine else "unknown"
        meta["header_lines"] = 2                    # title and author lines before the body
        meta["source"] = self.source_name
        meta["punctuation_removed"] = False
        return meta


@register("hindisamay", root="HindiSamay",
          description="Hindi prose from hindisamay.com online before 2022 (non-canonical authors, >= 500 words, not "
                      "purely factual), punctuation kept. File: <content id>_<title>.txt; metadata.csv has author, "
                      "genre, words, source URL. Not in git (do not redistribute): see the folder README.")
class HindiSamay(TitleAuthorCorpus):
    filename_regex = r"^(?P<content_id>\d+)_(?P<slug>.+)\.txt$"
    source_name = "hindisamay.com"
    int_fields = ("words", "id")
    STORY_GENRES = frozenset({"कहानी"})

    def _genre(self, fine: str) -> str:
        # Study genres: the site's कहानी (story) -> story; every other prose genre (आलोचना, लेख, निबंध, व्यंग्य,
        # संस्मरण, सिनेमा, विमर्श, यात्रावृत्त, लघुकथाएँ, ...) -> essay. meta["genre_fine"] keeps the site's label.
        return "story" if fine in self.STORY_GENRES else "essay"


@register("medium", root="Medium",
          description="English stories and non-factual essays from the fabiochiu/medium-articles dump, published "
                      "2014-2021, > 500 words. File: <publish-date>_<title-slug>.txt; metadata.csv has authors, "
                      "genre (story/essay), words, tags, URL. Not in git (do not redistribute): see the folder README.")
class Medium(TitleAuthorCorpus):
    filename_regex = r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<slug>.+)\.txt$"
    author_field = "authors"
    source_name = "medium.com"

    def _genre(self, fine: str) -> str:
        return fine if fine in ("story", "essay") else "unknown"

    def _meta(self, path: Path, text: str) -> dict[str, Any]:
        meta = super()._meta(path, text)
        if "date" in meta:
            meta["year"] = int(meta["date"][:4])
        if isinstance(meta.get("tags"), str):
            meta["tags"] = [t.strip() for t in meta["tags"].split(";") if t.strip()]
        return meta
