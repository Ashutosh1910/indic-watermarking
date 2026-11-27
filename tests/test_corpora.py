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


class TitleAuthorCorpusTest(unittest.TestCase):
    """hindisamay / medium layout: title, blank, author, blank, paragraphs; metadata.csv (with a BOM)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "00016_खुशबू.txt").write_text("खुशबू\n\nलेखक एक\n\nपहला अनुच्छेद है।", encoding="utf-8")
        (self.root / "00017_लेख.txt").write_text("एक लेख : उपशीर्षक\n\nलेखिका दो\n\nदूसरा पाठ।", encoding="utf-8")
        (self.root / "00018_बिना-मेटा.txt").write_text("बिना मेटा\n\nअनाम\n\nतीसरा पाठ।", encoding="utf-8")
        (self.root / "metadata.csv").write_text(
            "\ufefffile,id,title,author,genre,words\n00016_खुशबू.txt,16,खुशबू,लेखक एक,कहानी,2\n"
            "00017_लेख.txt,17,एक लेख : उपशीर्षक,लेखिका दो,आलोचना,2\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_hindisamay_metadata_and_genres(self):
        c = corpora.get("hindisamay", root=self.root)
        self.assertEqual(len(c), 3)                                  # metadata.csv is not a document
        story, essay, bare = c
        self.assertEqual((story.title, story.meta["author"], story.meta["words"], story.meta["id"]),
                         ("खुशबू", "लेखक एक", 2, 16))
        self.assertEqual((story.meta["genre"], story.meta["genre_fine"]), ("story", "कहानी"))
        self.assertEqual((essay.meta["genre"], essay.meta["genre_fine"]), ("essay", "आलोचना"))
        self.assertEqual(essay.title, "एक लेख : उपशीर्षक")
        # no metadata row: title and author from the header lines, genre unknown
        self.assertEqual((bare.title, bare.meta["author"], bare.meta["genre"]), ("बिना मेटा", "अनाम", "unknown"))
        self.assertEqual(story.text, "खुशबू\n\nलेखक एक\n\nपहला अनुच्छेद है।")     # raw text, header included
        self.assertEqual(story.meta["header_lines"], 2)


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

    def test_samalochan_genre_rule(self):
        c = corpora.get("samalochan")
        g = {d.meta["slug"]: d.meta["genre"] for d in c}
        self.assertEqual(set(g.values()), {"story", "essay"})
        self.assertTrue(all(v == "story" for k, v in g.items() if k.startswith("कथा-गाथा")))

    @unittest.skipUnless((corpora.DATA_ROOT / "HindiSamay").is_dir(), "HindiSamay not fetched (branch add-hindi-corpora)")
    def test_hindisamay(self):
        c = corpora.get("hindisamay")
        self.assertEqual(len(c), 1696)
        from collections import Counter
        self.assertEqual(Counter(d.meta["genre"] for d in c), {"story": 1035, "essay": 661})
        d = c[0]
        self.assertEqual(d.text.split("\n")[0], d.title)
        self.assertTrue(d.meta["author"] and d.meta["source_url"].startswith("https://www.hindisamay.com/"))

    @unittest.skipUnless((corpora.DATA_ROOT / "Medium").is_dir(), "Medium not fetched (branch add-hindi-corpora)")
    def test_medium(self):
        c = corpora.get("medium")
        self.assertEqual(len(c), 3992)
        from collections import Counter
        self.assertEqual(Counter(d.meta["genre"] for d in c), {"story": 1143, "essay": 2849})
        self.assertFalse(any(p.suffix != ".txt" for p in c.paths))        # filter_medium.py, metadata.csv skipped

    def test_aesthetics_duplicate_dropped(self):
        c = corpora.get("hindi-aesthetics")
        self.assertEqual(len(c), 968)
        self.assertEqual(len(c.duplicates), 1)
        self.assertTrue(any(not p.name.endswith(".txt") for p in c.paths))


if __name__ == "__main__":
    unittest.main()
