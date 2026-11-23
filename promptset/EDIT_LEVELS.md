# Edit-level prompt set (L1-L4, C, O, G_title)

`scripts/build_edit_prompts.py` turns **human documents** into model requests at graded levels of AI
involvement, for the SynthID-Text study (watermarked Qwen3-8B, English vs Hindi). Its rows have the same
schema as the G set (`build_promptset.py`), so one `generate.py` run handles everything, and
`trim.py` / `verify_generations.py` read the outputs unchanged. Config: `config/edit_levels.yaml`
(cutoffs, `tokens_per_word`, `min_frac` and `slack_words` come from `config/promptset.yaml`). Templates: the prompt kit
`config/ai_levels_prompts.json`, plus one template of our own for outline extraction (OX).
Tests: `python3 -m pytest promptset/tests/test_edit_prompts.py -q`.

## Levels

| level | model input | `{n}` / target | `max_new_tokens` | watermark |
|---|---|---|---|---|
| `L1_proofread`, `L2_polish`, `L3_paraphrase`, `L4_rewrite` | the whole cut human passage `{text}` | output is about as long as the input | 1.5 x input tokens, raised to the served cutoffs (below) | yes |
| `C_continue` (k = 75, 50, 25) | `{head}` = the first k % of the passage's words, cut at the sentence end nearest to k % | `{n}` = words left over (passage words - head words) | 1.3 x (passage tokens - head tokens), raised to the served cutoffs | yes |
| `OX_extract_outline` (stage 1, **ours**) | `{text}` = the passage; asks for 5-8 bullet points in the same language, in the model's own words | none | 512 (en) / 1024 (hi) | **no** |
| `O_outline` (stage 2) | `{points}` = the cleaned OX outline | `{n}` = passage words | 1.3 x passage tokens, raised to the served cutoffs | yes |
| `G_title` | `{title}` = the human doc's title; kit G templates 1 and 3 (indices 0 and 2) | `{n}` = passage words | 1.3 x passage tokens, raised to the served cutoffs | yes |

`{n}` is rounded to the nearest 10 (`lengths.round_n_to`); `target_words` holds the number in the prompt.

**`max_new_tokens` covers every cutoff the row serves.** A row's `cutoffs` can exceed its own passage: the
acceptance slack serves hi 1500 tokens from a 1363-token passage and 1200 words from 1170. A budget scaled on
the passage alone (1.3 x 1400 = 1820 < 1.25 x 1500) then cannot reach the largest cutoff it claims, which
`validate_promptset.py` rejects (`length.mnt_vs_token_cutoff`, `length.word_cutoff_unreachable`; 402 + 5
rows of the previous build, all C and G rows of `same_tokens` passages). So the builder takes

    max_new_tokens = ceil256( max( level factor x requested tokens,
                                   1.25 x (largest served token cutoff - head tokens),
                                   largest served word cutoff x passage tokens/word - head tokens ) )

with the validator's factor (1.25, `MNT_CUTOFF_FACTOR`; a test keeps the two equal) and the passage's own
tokens per word (`input_tokens / input_words`, as the validator uses); head tokens are 0 except for C rows.
Cutoffs are dropped instead only if the budget would push prompt + `max_new_tokens` past
`lengths.max_context` (32768, the validator's `--max-context`): the row then keeps the cutoffs the capped
budget reaches and the drop is counted under Skipped ("cutoffs dropped: over max_context"). No row of the
current build comes near it (passage + budget is at most about 16.5 k tokens). The O rows of `build-outline` use
the same rule. Steps are 256 tokens (`lengths.max_new_tokens_round`, was 64): generate.py batches only rows
with equal `max_new_tokens`, and 256 gives 38 distinct values instead of 136, so batches are fewer and
fuller. A budget is an upper bound (rows stop at EOS), so the extra headroom costs little.
The system prompt is the kit's, per language; every row carries `chat_template_kwargs: {enable_thinking:
false}`. The three kit phrasings of a level rotate across documents by a stable hash of (seed, level, doc
id): a document uses the same phrasing in both experiments and at every k. `template_id` is
`<level>/<index into the kit's list>`.

## Length rule and experiments

The rule, the same as `trim.py` uses for AI text: cut a text at the **last sentence end at or before N**
words or tokens, and accept it only if it keeps **n >= min(0.95 x N, N - slack)** ("95 % or one sentence
of slack"). Defaults from `config/promptset.yaml` (`min_frac: 0.95`, `slack_words: 30`): word cutoffs get
30 words of slack, token cutoffs 30 words' worth of tokens in the passage's language,
`round(30 x tokens_per_word[lang])` = 38 (en) / 137 (hi). The 95 % term decides long word cutoffs (800,
1200); the slack decides the short ones and every Hindi token cutoff up to 1500 (hi 1500 tokens: >= 1363).
To change it, set `lengths.slack_words` / `lengths.slack_tokens` (an int for every language) or
`lengths.min_frac` in `config/edit_levels.yaml`, or pass `build --slack-words S [--slack-tokens T]`;
**`--slack-words 0` (or `slack_words: 0`) gives the pure 95 % rule**. The rule used is in `build_meta.json`
(`acceptance_rule`, `lengths.slack`) and at the top of `edit_prompts_summary.md`. Use the same setting for
`trim.py` (its defaults are the same). The builder imports `trim.py`'s `trim_text`, `min_units`,
`resolve_slack` and `SENT_END` (sentence end = `. ! ? । ॥ …`, optionally followed by closing quotes or
brackets, then whitespace), so human inputs and AI outputs are cut by one rule.

Each human input is cut **before** editing, at the **largest** cutoff the document supports, so that one
generation per doc and level can later be prefix-trimmed at every smaller cutoff (the same trick as the G
set). Two passage variants per document (configurable under `experiments`):

* `same_words`: cut at the largest word cutoff (1200, else 800, ...);
* `same_tokens`: cut at the largest Qwen3-token cutoff (1500, else 800, ...).

For Hindi the two differ a lot (1200 words is about 5,500 tokens; 1500 tokens is about 310 words). When
both cuts give the same text (common for English), one set of rows is written with `experiment:
"same_words"` and `experiments: ["same_words", "same_tokens"]`. **Filter on `experiments`, not
`experiment`, when you analyse one experiment.** By default a document is used only if it yields a passage
for every experiment (`sampling.require_all_experiments`).

`cutoffs` in every row (and in `human_passages.jsonl`) lists the cutoffs the row serves: every configured
N at which the human passage itself trims to an accepted passage under the same rule. These are cutoffs up
to the passage length, minus those where no sentence end falls inside the acceptance window (see the
caveats). `trim.py gen` reads this field.

Token counts use the Qwen3 tokenizer (`tokenizer(text, add_special_tokens=False)`, as `trim.py` and the
verifier count them). `--tokenizer auto` (default) uses `/mnt/project-files/promptset_stage/qwen3_tokenizer`
when it exists. Without a tokenizer (`--tokenizer none`) tokens are estimated as words x
`tokens_per_word` and every row gets `"token_counts_estimated": true`; rebuild with the tokenizer before
the real run.

### Punctuation-stripped text (hindi-aesthetics)

`hindi-aesthetics` has its punctuation removed upstream and one sentence per line. For such sources
(`sources.<name>.punctuation_removed: true`, or a JSONL row's `punctuation_removed`), **every line break is
also a sentence end**; rows record `meta.punctuation_removed: true` and `sentence_end_rule:
"punct+linebreak"`. Proofreading (L1) or polishing (L2) punctuation-stripped text makes the model add
punctuation everywhere, which inflates the measured edit share, so **the default config excludes
hindi-aesthetics from L1 and L2** (`levels.<L>.punctuation_removed: exclude`; set it to `flag` to keep the
rows). L3, L4, C, O and G keep them.

`trim.py` honours each row's `sentence_end_rule` in both modes, so `trim.py human` on these passages and
`trim.py gen` on their C rows use the line-break rule too.

### Headers (samalochan)

samalochan posts start with headings, bylines, picture credits and `____` separators. With
`strip_header: true` the builder drops, before cutting, everything up to a separator in the first 8
non-empty lines, then leading lines of at most 10 words without sentence or clause punctuation (at most 8
lines in total). The passage is still a verbatim slice of the document (`meta.passage_char_start`).
Passages that still contain HTML remnants (`<![endif]>` and similar) are not used.

### Script check

A passage must have at least **0.8 Devanagari (hi) / 0.9 Latin (en)** among its letters, the threshold
`trim.py` applies to the same human passages (`DEFAULT_SCRIPT_MIN`; override per language with
`sampling.script_min`). A document whose passage fails is skipped like any other unusable document, so the
next document in the stable sample takes its place. The count is in `build_meta.json`
(`docs_skipped_script`, and the `(doc)` row "passage script share < 0.8" under `skips`) and in the summary.
In the current build this removes one samalochan document (`56e7c26ec9`, mostly Latin script, 0.72-0.75
Devanagari in its user turns), whose 15 rows the validator flagged (`script.hi_devanagari`).

### Titles (G_title)

`{title}` candidates per source come from `title_from`, tried in order, and must pass `title_check` (1-14
words, at least 90 % Devanagari letters for Hindi or Latin for English, no `: | _ / \`, not ending like a
sentence). For samalochan: the stripped heading lines, then the first line, then the file-name slug, which
is rejected when longer than 21 characters because samalochan cuts slugs short ("राहत इंदौरी एक ख़ुद्दार श").
Documents without a usable title get no G_title row. Every choice is listed in `titles_review.tsv`. A
heading line is sometimes the author's name, so check the list by hand and pass corrections with `--titles
FILE` (TSV `pair_id<TAB>title`, or JSONL `{"pair_id", "title"}`; an empty title drops the document).

## Inputs

* Registered corpora (`corpora` package): `--corpus samalochan --corpus hindi-aesthetics`. Language,
  license and the options above come from `sources.<name>` in the config. Genre is `meta.genre` if the
  corpus has one, else `"unknown"`.
* Generic JSONL (for example the English Medium corpus or HindiSamay): `--jsonl FILE` (repeatable), one
  document per line:

  ```json
  {"id": "m-000123", "language": "en", "title": "Why I Walk to Work", "text": "Full text ...",
   "genre": "personal", "license": "CC BY 4.0", "source": "medium"}
  ```

  `id`, `language` (`en` | `hi`) and `text` are required. `title` is needed for G_title, `genre` drives the
  stratified sample (`"unknown"` if missing), and `license` is copied into each row (otherwise
  `sources.<source>.license`). `source` defaults to `--jsonl-source` or the file name stem. Optional:
  `punctuation_removed` (bool) and `meta` (dict). Paragraphs are separated by newlines. Give the full text:
  the builder does the cutting. `pair_id` becomes `<source>/<id>`.

Options: `--n-docs` (per source and language; 0 = all), `--min-words` (minimum words of the source
document), `--levels` (subset; `O_outline` implies `OX_extract_outline`), `--experiments`, `--split`
(label, default `main`), `--pilot-docs K` (the first K sampled docs per source get `split: pilot`),
`--normalize NFC` (default: raw text, byte for byte).

Sampling is stable: documents are taken in the order of a seeded hash of their id, round-robin over genres,
skipping documents that fail `--min-words` or cannot be cut. A larger `--n-docs` keeps every document a
smaller one chose, with identical rows.

## Outputs (`--out`, default `promptset/data/edit/`)

| file | content |
|---|---|
| `edit_prompts.jsonl` | every row: L1-L4, C (one row per k), G_title, OX |
| `human_passages.jsonl` | one row per cut passage: `id` (`HP-...`), `pair_id`, `language`, `text`, `cutoffs`, `n_words`, `n_tokens`, `experiment(s)`, `input_cutoff_kind/cutoff`, `sentence_end_rule`, `meta`. The input of `trim.py human` and of the FPR check, identical to each row's `human_passage` |
| `titles_review.tsv` | the title chosen for every sampled document, or why there is none |
| `edit_prompts_summary.md` | counts per level x experiment x source x language, served cutoffs, `max_new_tokens` sums, skips |
| `build_meta.json` | arguments, lengths, tokenizer, counts, skips |
| `O_prompts.jsonl` | written by `build-outline` |

Row fields (the G-set schema plus some extra fields):
`id`, `pair_id` (human doc id), `language`, `level`, `genre`, `source` (corpus name), `source_id`, `license`, `split`,
`experiment`, `experiments`, `passage_id`, `input_cutoff_kind`, `input_cutoff`, `input_cut_by`, `input_words`,
`input_tokens` (of the cut passage), `token_counts_estimated`, `sentence_end_rule`, `human_passage` (the human
anchor for this row), `k` (C only), `target_words`, `max_new_tokens`, `cutoffs {words, tokens}`, `template_id`,
`messages [system, user]`, `chat_template_kwargs`, `generation {seed}` (sha256 of the id, as in the G set),
`watermark`, `meta` (`punctuation_removed`, `header_lines_stripped`, `passage_char_start`, `doc_words`,
`title_source`). The C rows also have `head`, `head_words`, `head_tokens`, `head_share` and
`continuation_*_human`. G_title rows have `title`. OX rows have `cutoffs` empty (outlines are not
scored) and the passage's cutoffs in `passage_cutoffs`. O rows have `points`, `n_bullets`,
`outline_source_id` and `outline_attempt`.

Ids: `<level>[_k<k>]-<source>-<sha1(pair_id)[:10]>-<sw|st>-<lang>`, for example
`C_continue_k50-samalochan-4842abd0ee-st-hi`. They are unique and stable across runs.

For C rows the generated `text` is only the continuation. The mixed text with an AI share of 1 - k is
`head` + `text`, joined by a newline for `punct+linebreak` sources and a space otherwise. That
joining belongs to the analysis or to `trim.py`, not to generate.py.

## `build-outline`

Reads `generate.py` outputs of the OX rows (each output row = input row + `text`, `finish_reason`,
`attempt`, ...). It keeps bullet lines only (`-`, `*`, `•`, `1.`, `1)`, `१.`), removes `**` markers, drops
the last bullet if the generation hit `max_new_tokens`, caps at 8 bullets and drops outlines with fewer than 3.
Per OX id the lowest attempt with a usable outline wins. It writes one `O_outline` row per OX row:
same `pair_id`, `experiment`, `passage_id`, `human_passage`, `cutoffs` = the passage's, and `{n}` = passage
words.

## Run order on the GPU

See `GPU_RUNBOOK.md` for setup, keys (`$RUNCFG`), sharding and regeneration. From the repo root:

```bash
E=promptset/data/edit; TOK=/path/to/Qwen3-8B     # or Qwen/Qwen3-8B, or the staged qwen3_tokenizer

# 1. build (CPU, ~1 min with the tokenizer)
python promptset/scripts/build_edit_prompts.py build --corpus samalochan --corpus hindi-aesthetics \
    --n-docs 150 --tokenizer $TOK --out $E            # + --jsonl medium_en.jsonl etc.

# 2. stage 1: outline extraction WITHOUT watermark (its own output file)
python promptset/scripts/generate.py --config $RUNCFG --prompts $E/edit_prompts.jsonl \
    --filter level=OX_extract_outline --no-watermark --out runs/edit_ox.jsonl

# 3. O rows from the outlines
python promptset/scripts/build_edit_prompts.py build-outline --gen runs/edit_ox.jsonl --out $E/O_prompts.jsonl

# 4. every other level, watermarked (one process per GPU with --shard i/N)
python promptset/scripts/generate.py --config $RUNCFG --prompts $E/edit_prompts.jsonl $E/O_prompts.jsonl \
    --filter watermark=true --filter language=hi --shard 0/4 --out runs/edit_hi.s0.jsonl

# 5. trim AI outputs (cutoffs from each row) and the human anchors (same passages, same rule)
python promptset/scripts/trim.py gen --in runs/edit_hi.s*.jsonl --out trimmed/edit_hi.jsonl --tokenizer $TOK
python promptset/scripts/trim.py human --in $E/human_passages.jsonl --out trimmed/edit_human_hi.jsonl \
    --tokenizer $TOK --words 100 200 400 800 1200 --tokens 200 400 800 1500

# 6. verify (TPR on edits, FPR on the human anchors, same keys)
python promptset/scripts/verify_generations.py --config runs/edit_hi.s0.config.yaml \
    --in trimmed/edit_hi.jsonl --out scores/edit_hi.jsonl
python promptset/scripts/verify_generations.py --config runs/edit_hi.s0.config.yaml \
    --in trimmed/edit_human_hi.jsonl --out scores/edit_human_hi.jsonl
```

Steps 2 and 4 use separate output files because generate.py refuses to mix watermark configs in one file.
Regenerate length failures from step 5 as in the runbook (`--only-ids ... --attempt k`).

## Current build (`promptset/data/edit/`, Qwen3 tokenizer, 150 docs per corpus)

4,597 rows from 600 passages (150 docs x 2 experiments x 2 corpora), built with the default rule (95 % or
one sentence of slack). No English source yet. `validate_promptset.py --strict` (Qwen3 tokenizer): 0
errors, 18 warnings (`balance.en_hi_edit`: no English rows yet).

| level | samalochan (sw / st) | hindi-aesthetics (sw / st) |
|---|---|---|
| L1_proofread | 150 / 150 | excluded |
| L2_polish | 150 / 150 | excluded |
| L3_paraphrase | 150 / 150 | 150 / 150 |
| L4_rewrite | 150 / 150 | 150 / 150 |
| C_continue k75, k50, k25 (each) | 150 / 149 | 150 / 150 |
| OX_extract_outline | 150 / 150 | 150 / 150 |
| G_title | 77 / 77 | 123 / 123 |

Sum of `max_new_tokens` is 15.8 M tokens (largest 9,984; 38 distinct values): an upper bound, and most
rows stop at EOS well before it. Of the +0.46 M over the previous build (15.3 M, ceil64, budgets on the
passage only), about 0.09 M is the cutoff floor above and the rest the 256 step.
`edit_prompts_summary.md` has the full tables.

## Caveats

* **Acceptance window at small cutoffs.** With the pure 95 % rule the window is 5 % of N: 5 words at
  N = 100, 10 tokens at N = 200. Hindi sentences are long (samalochan about 100-150 tokens), so with that
  rule the human anchor existed at 200 tokens for only 20/150 samalochan passages and at 100 words for
  33-35/150, 35 of 150 samalochan documents fell back from 1500 to a smaller token cutoff, and 35 documents
  (27 samalochan, 8 hindi-aesthetics) failed every token cutoff and were skipped, which favoured documents
  with shorter sentences. The default slack (30 words / 137 hi tokens) fixes most of this: in the current
  build the anchor exists at 200 tokens for 132/150 samalochan passages and at 100 words for 127-129/150,
  20/150 samalochan documents fall back from 1500 tokens, and 1 document is skipped for length (1 more for
  script, see above). The cost is that a
  short-cutoff passage can be much shorter than N (hi 200 tokens accepts from 63 tokens): analyse
  `n_tokens` / `n_words`, not only N. Rebuild with `--slack-words 0` to compare with the pure rule.
* **Low-Devanagari documents.** The slack admits documents the old rule skipped; such a document is now
  caught by the passage script check (see "Script check") and replaced by the next one in the sample.
* Hindi same_words rows are long: about 5,500 input tokens and `max_new_tokens` up to 9,984 for
  L1-L4.
* `edit_prompts.jsonl` is about 90 MB, because the Hindi passage appears in `human_passage` and in the
  prompt. Keep it out of git.
