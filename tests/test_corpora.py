"""Tests for corpora/. Run from the repo root:  python3 -m unittest discover tests"""
from __future__ import annotations

import tempfile
import unicodedata
import unittest
from pathlib import Path

import corpora
from corpora import registry
from corpora.base import TextDirCorpus

NUKTA_PRE = "क़"                                   # क़ as one code point (not NFC)
NUKTA_NFC = unicodedata.normalize("NFC", NUKTA_PRE)     # क + ़


class TempCorpusTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "b.txt").write_text("दूसरा  पाठ‍", encoding="utf-8")
        (self.root / "a-story").write_text(f"पहला {NUKTA_PRE} पाठ है", encoding="utf-8")   # no extension
        (self.root / "copy.txt").write_text(f"पहला {NUKTA_PRE} पाठ है", encoding="utf-8")  # duplicate of a-story
        (self.root / ".hidden").write_text("x", encoding="utf-8")
        self._saved = dict(registry._REGISTRY)

    def tearDown(self):
        registry._REGISTRY.clear()
        registry._REGISTRY.update(self._saved)
        self.tmp.cleanup()

    def test_raw_text_is_untouched_by_default(self):
        cls = registry.register_text_dir("tmp-raw", self.root)
        texts = {d.meta["filename"]: d.text for d in corpora.get("tmp-raw")}
        self.assertEqual(texts["b.txt"], "दूसरा  पाठ‍")       # double space and ZWJ kept
        self.assertIn(NUKTA_PRE, texts["a-story"])
        self.assertIs(cls.name, "tmp-raw")

    def test_dedupe_hidden_files_and_extensionless(self):
        registry.register_text_dir("tmp-d", self.root)
        c = corpora.get("tmp-d")
        self.assertEqual([p.name for p in c.paths], ["a-story", "b.txt"])
        self.assertEqual({p.name: o.name for p, o in c.duplicates.items()}, {"copy.txt": "a-story"})
        self.assertEqual(len(corpora.get("tmp-d", dedupe=False)), 3)

    def test_normalize_is_opt_in(self):
        registry.register_text_dir("tmp-n", self.root)
        doc = corpora.get("tmp-n", normalize="NFC")[0]
        self.assertNotIn(NUKTA_PRE, doc.text)
        self.assertIn(NUKTA_NFC, doc.text)
        self.assertEqual(corpora.get("tmp-n", normalize=str.upper)[1].text, "दूसरा  पाठ‍".upper())
        with self.assertRaises(ValueError):
            corpora.get("tmp-n", normalize="NFX")

    def test_filename_regex_metadata_and_min_words(self):
        registry.register_text_dir("tmp-r", self.root, filename_regex=r"^(?P<letter>[a-z])[.-]")
        c = corpora.get("tmp-r", min_words=3)
        self.assertEqual([d.meta.get("letter") for d in c], ["a"])

    def test_subclass_registration(self):
        @corpora.register("tmp-cls", root=self.root, description="desc")
        class Tmp(TextDirCorpus):
            pattern = "*.txt"

            def _meta(self, path, text):
                return {"n_chars": len(text)}

        c = corpora.get("tmp-cls")
        self.assertEqual(len(c), 2)                # b.txt, copy.txt
        self.assertEqual(c[0].meta["n_chars"], len(c[0].text))
        self.assertEqual(corpora.describe()["tmp-cls"], "desc")
        with self.assertRaises(ValueError):
            corpora.register("tmp-cls", root=self.root)(Tmp)

    def test_unknown_dataset_and_missing_root(self):
        with self.assertRaises(KeyError):
            corpora.get("does-not-exist")
        registry.register_text_dir("tmp-missing", self.root / "nope")
        with self.assertRaises(FileNotFoundError):
            corpora.get("tmp-missing")

    def test_loader_batches_shuffle_transform_filter(self):
        registry.register_text_dir("tmp-l1", self.root)
        registry.register_text_dir("tmp-l2", self.root, pattern="*.txt", exclude={"copy.txt"})
        loader = corpora.CorpusLoader(["tmp-l1", "tmp-l2"], batch_size=2)
        self.assertEqual(len(loader), 2)
        self.assertEqual([[d.id for d in b] for b in loader],
                         [["tmp-l1/a-story", "tmp-l1/b.txt"], ["tmp-l2/b.txt"]])
        ids = lambda l: [d.id for d in l]
        a = corpora.CorpusLoader(["tmp-l1", "tmp-l2"], shuffle=True, seed=3)
        b = corpora.CorpusLoader(["tmp-l1", "tmp-l2"], shuffle=True, seed=3)
        self.assertEqual(ids(a), ids(b))
        self.assertEqual(sorted(ids(a)), sorted(ids(corpora.CorpusLoader(["tmp-l1", "tmp-l2"]))))
        self.assertEqual(list(corpora.CorpusLoader("tmp-l1", transform=lambda d: d.n_words)), [4, 2])
        self.assertEqual(len(list(corpora.CorpusLoader("tmp-l1", filter=lambda d: "‍" in d.text))), 1)
        self.assertEqual(len(list(corpora.CorpusLoader("tmp-l1", batch_size=3, drop_last=True))), 0)


@unittest.skipUnless((corpora.DATA_ROOT / "Samalochan").is_dir(), "real data not present")
class RealDataTest(unittest.TestCase):
    def test_registered(self):
        self.assertTrue({"hindi-aesthetics", "samalochan"} <= set(corpora.available()))

    def test_samalochan(self):
        c = corpora.get("samalochan")
        self.assertEqual(len(c), 576)
        d = c[0]
        self.assertRegex(d.meta["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertLess(max(x.meta["year"] for x in c[-5:]), 2022)

    def test_aesthetics_duplicate_dropped(self):
        c = corpora.get("hindi-aesthetics")
        self.assertEqual(len(c), 968)
        self.assertEqual(len(c.duplicates), 1)
        self.assertTrue(any(not p.name.endswith(".txt") for p in c.paths))


if __name__ == "__main__":
    unittest.main()
