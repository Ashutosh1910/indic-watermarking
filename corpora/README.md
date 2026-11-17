# corpora: one entry point for the text under `data/`

A small, dependency-free package (Python 3.10+, standard library only) that serves every corpus in `data/`
as `Document` records with metadata. It never writes to `data/`, and it returns text byte-for-byte unless you
ask for normalisation. That keeps ZWJ, NBSP, double spaces and precomposed nukta available for drift tests.

```python
import corpora

corpora.available()                     # ['hindi-aesthetics', 'samalochan']
sam = corpora.get("samalochan")         # Sequence[Document]: len(), indexing, slicing, iteration
doc = sam[0]
doc.id, doc.title, doc.text, doc.meta   # meta: date, year, slug, first_line, source, ...

# Several corpora, batched and shuffled (reproducible per seed and epoch)
for batch in corpora.CorpusLoader("all", batch_size=16, shuffle=True, seed=0):
    ...

# Opt-in normalisation, length filter, transform and filter hooks
loader = corpora.CorpusLoader(["samalochan"], normalize="NFC", min_words=1000,
                              filter=lambda d: d.meta["year"] < 2020, transform=lambda d: d.text)
```

From the command line, run in the repo root:

```bash
python3 -m corpora list
python3 -m corpora stats
python3 -m corpora show samalochan 5
```

A `Corpus` is a map-style dataset (`__len__` and `__getitem__`), so it can go straight into
`torch.utils.data.DataLoader(corpus, batch_size=..., collate_fn=list)` if you need workers. The package itself
does not import torch.

## Datasets

| name | folder | documents | notes |
|---|---|---|---|
| `hindi-aesthetics` | `data/Hindi-Aesthetics-Corpus` | 968 | Every file is read, with or without `.txt`. One exact duplicate (`आलमे-तसबीर-1` = `आकाशबाणी`) is dropped. Punctuation was removed upstream, so sentences are one per line and double spaces are common (`meta["punctuation_removed"] = True`). 491 files use precomposed nukta, which is not NFC. |
| `samalochan` | `data/Samalochan` | 576 | `date`, `year` and `slug` come from `<date>_<slug>.txt`. The title is the slug. `first_line` is the first line of the post, which is sometimes a section label rather than the headline. |

## Options (every corpus)

| option | default | effect |
|---|---|---|
| `normalize` | `None` | `None` returns raw text. It also accepts `"NFC"`, `"NFD"`, `"NFKC"`, `"NFKD"`, or any `str -> str` callable. |
| `dedupe` | `True` | Drops files whose bytes match an earlier file. Dropped files are listed in `corpus.duplicates`. |
| `min_words` | `0` | Skips documents with fewer whitespace-separated words. |
| `root` | registered folder | Points a registered corpus at another copy of the data. |

## Adding a dataset

Put the files under `data/<Folder>/` and register it in `corpora/datasets.py`. For a plain folder of text
files, one call is enough:

```python
from .registry import register_text_dir

register_text_dir("hindi-wiki", "Hindi-Wikipedia", description="...", pattern="*.txt",
                  filename_regex=r"^(?P<page_id>\d+)_")       # named groups become metadata
```

For custom titles, metadata, file selection or decoding, subclass `TextDirCorpus` and override `_paths`,
`_read`, `_title` or `_meta` (see `HindiAesthetics` and `Samalochan` in `datasets.py`). For a format that
is not one file per document, such as a single JSONL file, subclass `Corpus` and override `paths` and
`load` so that each index maps to one record.

Run the tests with `python3 -m unittest discover tests`.
