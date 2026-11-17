# Retokenization drift on Hindi text, with an English baseline (tokenizer only)

**Question:** a SynthID-Text detector never sees the model's token IDs. It sees *text*, which it has to
tokenize again. How many of the original tokens, and of the 5-token windows that carry the watermark,
survive that round trip once the text has gone through the things that happen to real text:
copy-paste, Unicode normalisation, zero-width characters being stripped, whitespace clean-up?

This version needs **no model and no GPU**. It uses only the Qwen3 tokenizer (a few MB, shared by every
Qwen3 size including Qwen3-8B) and real human-written Hindi from a public dataset. An English arm runs
the same perturbations and metrics on the English side of the same corpus, so Hindi can be read against
a baseline. It measures the
retokenization gap only; watermark z-scores are out of scope.

Everything stays in this folder: the virtual env (`.venv/`), the Hugging Face cache (`.hf-cache/`), the
downloaded text (`texts/`) and the results (`outputs/`). None of them are committed.

## Setup (macOS or Linux)

```bash
cd retokenise-experiment
./setup.sh                  # creates .venv/ and installs requirements.txt into it (no torch)
```

## Run

```bash
./run_all.sh --quick        # 10 passages, a few seconds after the first download
./run_all.sh                # 100 passages of ~200 tokens, Hindi
./run_all.sh --lang en      # the same for English
./run_all.sh --lang both --n-passages 347   # both languages + outputs/compare.md side by side
```

`--lang both` writes `outputs/hi/`, `outputs/en/` and `outputs/compare.md`. 347 is the largest matched
size: the English side of IITB test+validation (3,027 records) yields 347 passages of 200 tokens, the
Hindi side about 1,195, because Qwen3 needs roughly 4× more tokens per character for Devanagari. Fetch
the full splits first with `cd src && ../.venv/bin/python fetch_texts.py --lang en --n-lines 5000` (and
the same for `hi`); the default of 3,000 records gives slightly fewer English passages. `drift.py`
prints a warning when it gets fewer passages than asked for.

On the first run, `run_all.sh` downloads the Hindi text with `src/fetch_texts.py` and the tokenizer
from Hugging Face. Other options:

```bash
./run_all.sh --tokenizer Qwen/Qwen3-8B --n-passages 300 --max-tokens 200 --out outputs/run2
./run_all.sh --perturbations none NFKC strip_zw webform
./run_all.sh --texts /path/to/indiccorp_hi_sample.txt --out outputs/indiccorp
```

## Data source

The default is the **Hindi side of the IIT Bombay English-Hindi parallel corpus**
([`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi)), test and
validation splits, field `translation.hi`, first 3,000 records. It is human-written and translated news,
released in 2018, so it predates LLM-generated text. The text is kept byte-for-byte; nothing is normalised
before the experiment starts.

The **English arm** uses the English side of the same corpus (`--lang en`, field `translation.en`). Because
IITB is parallel, both arms share content, period and pre-LLM provenance; one side of each pair is a human
translation of the other. The exact source is written to `texts/hi.source.json` and repeated at the
top of every `summary.md`.

To use another dataset:

```bash
cd src
../.venv/bin/python fetch_texts.py --dataset wikimedia/wikipedia --config 20231101.hi \
    --splits train --field text --n-lines 2000
```

For **IndicCorp** (downloaded separately from AI4Bharat), take a slice of its plain-text file:
`head -n 3000 hi.txt > texts/hi.txt`. Pre-LLM sources such as IITB and IndicCorp are better than
recent web text, because newer pages may already contain machine-generated text.

## Method

Short records are joined into passages and cut at a character boundary (never inside a character)
so each one encodes to at most `--max-tokens` tokens. For each passage *x*:

```
ids  = encode(x)                  # the tokenizer's own canonical split of human text
ids2 = encode(perturb(decode(ids)))
```

| perturbation | what it models |
|---|---|
| `none` | a plain decode → encode round trip (the control) |
| `NFC`, `NFD`, `NFKC` | Unicode normalisation by editors, browsers, web forms or databases |
| `strip_zw` | sanitisers that drop ZWJ / ZWNJ / ZWSP / BOM (ZWJ and ZWNJ change how Devanagari conjuncts render) |
| `whitespace` | NBSP and other unusual spaces turned into normal spaces, runs collapsed, ends trimmed |
| `nukta_precomposed` | क़ ख़ ग़ ज़ ड़ ढ़ फ़ य़ written as one code point (U+0958–095F), as some keyboards do |
| `webform` | NFKC + `strip_zw` + `whitespace` together, a plausible worst case for pasted text |

Metrics, comparing `ids` with `ids2`:

- **exact match**: the two ID lists are identical;
- **aligned positions**: share of the original positions inside `difflib.SequenceMatcher` matching blocks;
- **5-gram survival**: share of the original 5-token windows (as a multiset) that still exist after
  re-encoding. SynthID with `ngram_len=5` derives each position's g-value from tokens *t−4…t*, so this is
  the share of positions that would keep their watermark signal. **This is the number that matters.**

## Outputs (`outputs/<run>/`)

| file | contents |
|---|---|
| `summary.md` | two tables per perturbation. **All passages**: how often the text changed, exact match, aligned positions, 5-gram survival and change in token count. **Changed passages only**: the cost each time the perturbation actually changes something |
| `mismatches.md` | the most frequent mismatched spans, with their code points, original tokens → re-encoded tokens and context |
| `mismatches.csv` | the same spans as CSV (read by `compare.py`) |
| `per_passage.csv` | every metric for each passage and perturbation |
| `../compare.md` | with `--lang both`: fertility of each language, both tables as hi / en cells, top mismatches per language |

## Results (Qwen3 tokenizer, IITB test+validation, 347 passages per language)

The full table is in [`results/hi_vs_en.md`](results/hi_vs_en.md). In short:

- **Clean human text barely drifts in either language.** The `none` control is 100% in both, and every
  perturbation keeps at least 99.1% of 5-gram windows when averaged over all passages.
- **Qwen3's built-in NFC undoes NFC, NFD and precomposed nukta.** 37% of Hindi passages contain a
  nukta letter that `nukta_precomposed` rewrites, and every one of them re-encodes to the same IDs.
- **Whitespace clean-up is the only perturbation that fires often, and it fires far more in Hindi.** The
  Hindi side of IITB has many double spaces, so collapsing them changes 19.9% of Hindi passages and 2.3%
  of English ones. When it fires the cost is similar (95.4% vs 96.5% 5-gram survival), since each
  removed space is one token in both languages.
- **NFKC only hit English**, through NBSPs (U+00A0) in the source; the Hindi side has none.
- **ZWJ/ZWNJ never occur** in either side of this corpus, so `strip_zw` is untested here. Web-scraped Hindi
  (IndicCorp, Wikipedia) is the place to measure it.
- Fertility differs a lot: 1.09 chars/token and 4.7 tokens/word for Hindi, against 4.72 and 1.27 for English.

## What to expect, and caveats

- **`none` should be ~100%.** Human text encoded by the tokenizer is already its canonical split, so
  a plain round trip changes nothing. That is the point of the control: when watermarked *model* output
  drifts more than this, the extra drift comes from the model emitting non-canonical token sequences
  (or broken UTF-8 byte tokens), not from the tokenizer.
- **The tokenizer may undo some perturbations itself.** Qwen's tokenizer applies its own Unicode
  normaliser inside `encode()` (shown at the top of `summary.md`). If that is NFC, the NFC, NFD and
  nukta rows should show no drift, because the tokenizer normalises the text back before splitting it.
  That is a finding in itself: those edits are harmless for this tokenizer, but `NFKC`, `strip_zw` and
  `whitespace` are not undone.
- **Many perturbations rarely fire on clean text.** Clean news text seldom contains ZWJ or NBSP, so use
  the "changed passages only" table for the cost per occurrence, and the "all passages" table for how
  often it matters in practice.
- **This only covers this tokenizer.** Gemma, Llama and other tokenizers split Devanagari differently.
  Re-run with `--tokenizer` to compare.
- **Survival is not the same as detectability.** 5-gram survival is the share of watermark-carrying
  positions kept. How much that lowers a detection score also depends on text length and the detector,
  which this experiment deliberately leaves out.

## Web-crawled Hindi (ZWJ/ZWNJ check)

The IITB sides contain no zero-width characters, so a small web sample was streamed to test `strip_zw`:
1,500 documents each from mC4 Hindi (`allenai/c4`, config `hi`, Common Crawl up to 2020) and AI4Bharat
Sangraha (`verified/hin`). The commands are in `fetch_texts.py --help`. For mC4, only documents whose letters are
at least 80% Devanagari were kept (775 of 1,500; the rest are mostly Malayalam and Latin noise). Results are in
[`results/hi_web.md`](results/hi_web.md):

- **mC4 has ZWJ in 13% of documents**, almost always as virama + ZWJ + consonant (`क्‍या`, `दिल्‍ली`), the
  explicit half-form some Hindi input tools insert. Stripping it changes 7.7% of passages, and a changed
  passage keeps 93.4% of its 5-gram windows. Each ZWJ is split into two byte tokens, and removing it
  re-merges the neighbouring bytes, so one invisible character costs several windows.
- **Sangraha has none at all**, nor NBSP or BOM: its cleaning pipeline appears to strip them. Like IITB news, its
  main drift is whitespace collapse (17% of passages).
