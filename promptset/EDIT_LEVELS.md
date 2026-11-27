# Edit-level prompt set (L1-L4, C, O, G_title)

`scripts/build_edit_prompts.py` turns **human documents** into model requests at graded levels of AI
involvement, for the SynthID-Text study (watermarked Qwen3-8B, English vs Hindi). Its rows have the same
schema as the G set (`build_promptset.py`), so one `generate.py` run handles everything, and
`trim.py` / `verify_generations.py` read the outputs unchanged. The base texts come only from our study
corpora (Hindi: HindiSamay + Samalochan; English: Medium), sampled with the same story/essay mix per
language; the prompt wording comes only from the fixed kit templates. Config: `config/edit_levels.yaml`
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

`hindi-aesthetics` is **not a study corpus** and is not in the default build (since the genre-matched
build); the rules below apply when it is named with `--corpus hindi-aesthetics` or for any other source
marked `punctuation_removed`. It has its punctuation removed upstream and one sentence per line. For such sources
(`sources.<name>.punctuation_removed: true`, or a JSONL row's `punctuation_removed`), **every line break is
also a sentence end**; rows record `meta.punctuation_removed: true` and `sentence_end_rule:
"punct+linebreak"`. Proofreading (L1) or polishing (L2) punctuation-stripped text makes the model add
punctuation everywhere, which inflates the measured edit share, so **the default config excludes
hindi-aesthetics from L1 and L2** (`levels.<L>.punctuation_removed: exclude`; set it to `flag` to keep the
rows). L3, L4, C, O and G keep them.

`trim.py` honours each row's `sentence_end_rule` in both modes, so `trim.py human` on these passages and
`trim.py gen` on their C rows use the line-break rule too.

### Headers (samalochan, hindisamay, medium)

samalochan posts start with headings, bylines, picture credits and `____` separators. With
`strip_header: true` the builder drops, before cutting, everything up to a separator in the first 8
non-empty lines, then leading lines of at most 10 words without sentence or clause punctuation (at most 8
lines in total). hindisamay and medium files have a fixed header (title, blank line, author, blank line),
and titles can be long (649 Medium titles have more than 10 words), so for them `header_lines: 2` drops the
title and author lines unconditionally first; the heuristic then drops any further heading lines (a date,
a publication name, "भूमिका", a translator's note such as "(डोगरी से अनुवाद : …)"). The passage is
still a verbatim slice of the document (`meta.passage_char_start`; `meta.header_lines_stripped` counts
the dropped lines). Passages that still contain HTML remnants (`<![endif]>` and similar) are not used.

### Script check

A passage must have at least **0.8 Devanagari (hi) / 0.9 Latin (en)** among its letters, the threshold
`trim.py` applies to the same human passages (`DEFAULT_SCRIPT_MIN`; override per language with
`sampling.script_min`). A document whose passage fails is skipped like any other unusable document, so the
next document in the stable sample takes its place. The count is in `build_meta.json`
(`docs_skipped_script`, and the `(doc)` row "passage script share < 0.8" under `skips`) and in the summary.
In the build that introduced the check this removed one samalochan document (`56e7c26ec9`, mostly Latin
script, 0.72-0.75 Devanagari in its user turns), whose 15 rows the validator flagged
(`script.hi_devanagari`). The current build skips one samalochan document for script.

### Titles (G_title)

`{title}` candidates per source come from `title_from`, tried in order, and must pass `title_check` (1-14
words, at least 90 % Devanagari letters for Hindi or Latin for English, no `: | _ / \`, not ending with a
colon or like a sentence; a trailing `...` / `…` is allowed). hindisamay and medium use the metadata title
(`title_from: [title]`) and allow a colon inside it (`sources.<name>.title_check`), because there it marks a
subtitle ("कबीर : एक अंतहीन यात्रा", "The Alchemist: A book review"), not a section label. For samalochan: the stripped heading lines, then the first line, then the file-name slug, which
is rejected when longer than 21 characters because samalochan cuts slugs short ("राहत इंदौरी एक ख़ुद्दार श").
Documents without a usable title get no G_title row. Every choice is listed in `titles_review.tsv`. A
heading line is sometimes the author's name, so check the list by hand and pass corrections with `--titles
FILE` (TSV `pair_id<TAB>title`, or JSONL `{"pair_id", "title"}`; an empty title drops the document).

## Study corpora

| language | corpus (`--corpus` name) | folder | documents | genres (`meta.genre`) | license (copied into each row) |
|---|---|---|---|---|---|
| hi | `hindisamay` | `data/HindiSamay` | 1,696 | story 1,035 / essay 661 | hindisamay.com (MGAHV), published with the authors' permission; copyright stays with the authors; non-commercial research only, do not redistribute |
| hi | `samalochan` | `data/Samalochan` | 576 | story 36 / essay 540 (rule below) | samalochan.com posts; copyright of the authors; research use only, do not redistribute |
| en | `medium` | `data/Medium` | 3,992 | story 1,143 / essay 2,849 | Medium posts via fabiochiu/medium-articles (dump labelled MIT); copyright stays with each Medium author; non-commercial research only, do not redistribute |

`hindi-aesthetics` (`data/Hindi-Aesthetics-Corpus`) stays registered but is not a study corpus: its
punctuation was stripped upstream.

**Fetching HindiSamay and Medium.** Their texts are not in this branch: copyright stays with the authors
and their READMEs say not to redistribute them. They live on branch `add-hindi-corpora` (commit
`2f2abaf`), with a `metadata.csv` and a README each (Medium also has `filter_medium.py`, which rebuilds
the set from the Hugging Face dump). From the repo root:

```bash
git fetch origin add-hindi-corpora
git checkout FETCH_HEAD -- data/HindiSamay data/Medium      # puts the files in the working tree (staged)
git restore --staged data/HindiSamay data/Medium             # unstage: keep them out of commits
```

`/data/HindiSamay/` and `/data/Medium/` are in the repo-root `.gitignore`, so they stay untracked. Do not
commit them, and keep `edit_prompts.jsonl` / `human_passages.jsonl` (which embed the texts) out of git too.

**Genres.** The study compares two coarse genres, `story` and `essay`; every row has `genre` set to one of
them and keeps the corpus's own label in `meta.genre_fine` (or how it was assigned in `meta.genre_rule`).

* hindisamay: the site's `कहानी` -> story; every other prose genre (आलोचना, लेख, सिनेमा, विमर्श, निबंध,
  व्यंग्य, संस्मरण, यात्रावृत्त, वैचारिकी, डायरी, बात-चीत, लघुकथाएँ, व्याख्यान, आत्मकथ्य, पत्र) -> essay.
  The 6 `लघुकथाएँ` (short-short story) texts are essays under this rule.
* medium: `metadata.csv` already has story / essay (from the authors' Medium tags).
* samalochan: the site has no genre field and nothing in `data/Samalochan` gives one, so a fixed rule
  decides (`corpora/datasets.py`, `samalochan_genre`): **story** if the file slug starts with `कथा-गाथा`
  (samalochan's fiction column, 33 posts) or one of the first three non-empty lines is exactly a section
  label `कहानी` / `कहानियाँ` / `लम्बी कहानी` (3 more); **essay** otherwise. Essay is a residual class:
  criticism, essays, memoirs, interviews and reviews, plus the few stories posted with only an editor's
  note (for example "युवा कथाकार … की नई कहानी"), which no reliable rule separates from criticism about a
  story. The default build takes no stories from samalochan (see Sampling), so this matters only for
  which samalochan posts can be essays.

## Sampling

**genre_mix (the default, the study build).** `build` without `--corpus` / `--jsonl` loads every source
named under `sampling.genre_sources` and draws, **per language**, `sampling.n_docs_per_language` (300)
documents split by `sampling.genre_mix` (story 0.5 / essay 0.5), so English and Hindi have the same
story/essay mix. Each genre's quota is split over that language's sources by the shares in
`sampling.genre_sources`:

| language | story (150) | essay (150) |
|---|---|---|
| hi | hindisamay 150 | hindisamay 75 + samalochan 75 |
| en | medium 150 | medium 150 |

Stories come only from HindiSamay: samalochan's story label is a heuristic and its fiction-column posts
usually open with an author bio. Quotas are integers by largest remainder (ties by name). Inside a
(genre, source) cell documents are taken in the order of a seeded stable hash of their id (seed
`20261123`, the same order as per_corpus mode), skipping documents that fail `--min-words`, the length rule
for every experiment (`require_all_experiments`) or the script check. With the default shares and no
top-ups, a larger `--n-docs` keeps every document a smaller one chose (each cell's picks are a prefix of
its order). If a source runs out of usable documents, the other sources of the same genre
fill its quota (largest share first); if a genre cannot be filled the build warns. Genres outside
`genre_mix` are never sampled. The sample is written interleaved over the cells, so `--pilot-docs K` (the
first K documents per language) keeps the mix. `--n-docs N` sets the number per language.

**per_corpus.** With `--corpus NAME` (repeatable) or `--jsonl FILE`, each source is sampled on its own:
`--n-docs` (default `sampling.n_docs`, 150; 0 = all) documents per source and language, round-robin over
genres (`sampling.stratify_by`), as before the genre-matched build. `--sampling genre_mix` pools the named
inputs per language with the genre mix instead (shares renormalised over the loaded sources; a genre with
no configured share is split equally).

## Inputs

* Registered corpora (`corpora` package): `hindisamay`, `samalochan`, `medium` by default, or any
  registered corpus with `--corpus NAME`. Language, license and the options above come from
  `sources.<name>` in the config. Genre is `meta.genre` (`"unknown"` if the corpus has none).
* Generic JSONL: `--jsonl FILE` (repeatable), one document per line:

  ```json
  {"id": "m-000123", "language": "en", "title": "Why I Walk to Work", "text": "Full text ...",
   "genre": "essay", "license": "CC BY 4.0", "source": "mysource"}
  ```

  `id`, `language` (`en` | `hi`) and `text` are required. `title` is needed for G_title, `genre` drives the
  stratified sample and the genre mix (`"unknown"` if missing), and `license` is copied into each row
  (otherwise `sources.<source>.license`). `source` defaults to `--jsonl-source` or the file name stem.
  Optional: `punctuation_removed` (bool) and `meta` (dict). Paragraphs are separated by newlines. Give the
  full text: the builder does the cutting. `pair_id` becomes `<source>/<id>`.

Options: `--sampling genre_mix|per_corpus`, `--n-docs` (per language in genre_mix, per source in
per_corpus), `--min-words` (minimum words of the source document), `--levels` (subset; `O_outline`
implies `OX_extract_outline`), `--experiments`, `--split` (label, default `main`), `--pilot-docs K`,
`--normalize NFC` (default: raw text, byte for byte).

## Outputs (`--out`, default `promptset/data/edit/`)

| file | content |
|---|---|
| `edit_prompts.jsonl` | every row: L1-L4, C (one row per k), G_title, OX |
| `human_passages.jsonl` | one row per cut passage: `id` (`HP-...`), `pair_id`, `language`, `text`, `cutoffs`, `n_words`, `n_tokens`, `experiment(s)`, `input_cutoff_kind/cutoff`, `sentence_end_rule`, `meta`. The input of `trim.py human` and of the FPR check, identical to each row's `human_passage` |
| `titles_review.tsv` | the title chosen for every sampled document (with its genre), or why there is none |
| `edit_prompts_summary.md` | documents per language x genre x source (quota, sampled), rows per level x language x genre x experiment and per source, served cutoffs (per source and per-language coverage), `max_new_tokens` sums per level and per language, skips |
| `build_meta.json` | arguments, lengths, tokenizer, counts, skips |
| `O_prompts.jsonl` | written by `build-outline` |

Row fields (the G-set schema plus some extra fields):
`id`, `pair_id` (human doc id), `language`, `level`, `genre`, `source` (corpus name), `source_id`, `license`, `split`,
`experiment`, `experiments`, `passage_id`, `input_cutoff_kind`, `input_cutoff`, `input_cut_by`, `input_words`,
`input_tokens` (of the cut passage), `token_counts_estimated`, `sentence_end_rule`, `human_passage` (the human
anchor for this row), `k` (C only), `target_words`, `max_new_tokens`, `cutoffs {words, tokens}`, `template_id`,
`messages [system, user]`, `chat_template_kwargs`, `generation {seed}` (sha256 of the id, as in the G set),
`watermark`, `meta` (`punctuation_removed`, `header_lines_stripped`, `passage_char_start`, `doc_words`,
`genre_fine` or `genre_rule`, `title_source`). The C rows also have `head`, `head_words`, `head_tokens`, `head_share` and
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

# 0. fetch HindiSamay and Medium into data/ (see "Study corpora"; not committed)
# 1. build (CPU, ~2 min with the tokenizer): 300 docs per language, story/essay 50/50
python promptset/scripts/build_edit_prompts.py build --tokenizer $TOK --out $E
#    (the current build used TOK=/mnt/project-files/promptset_stage/qwen3_tokenizer/)

# 2. stage 1: outline extraction WITHOUT watermark (its own output file)
python promptset/scripts/generate.py --config $RUNCFG --prompts $E/edit_prompts.jsonl \
    --filter level=OX_extract_outline --no-watermark --out runs/edit_ox.jsonl

# 3. O rows from the outlines
python promptset/scripts/build_edit_prompts.py build-outline --gen runs/edit_ox.jsonl --out $E/O_prompts.jsonl

# 4. every other level, watermarked (one process per GPU with --shard i/N)
python promptset/scripts/generate.py --config $RUNCFG --prompts $E/edit_prompts.jsonl $E/O_prompts.jsonl \
    --filter watermark=true --filter language=hi --shard 0/4 --out runs/edit_hi.s0.jsonl
#    (and the same with --filter language=en --out runs/edit_en.s0.jsonl ...)

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

## Current build (`promptset/data/edit/`, Qwen3 tokenizer, genre-matched, 300 docs per language)

Built with `build --tokenizer /mnt/project-files/promptset_stage/qwen3_tokenizer/ --out promptset/data/edit`
and the default rule (95 % or one sentence of slack): **10,455 rows** (en 5,120, hi 5,335) from 1,173 passages
of 600 documents. `validate_promptset.py --strict` (Qwen3 tokenizer): **0 errors, 7 warnings**, all
`balance.en_hi_edit`: G_title in 4 cells (titles usable for en 280 / hi 269 documents) and C_continue
hi story same_tokens 149 vs 150 (one HindiSamay passage has no inner sentence end) in 3 cells. Every
other level x genre x experiment cell has equal en and hi counts.

Documents (all quotas filled):

| language | genre | source | documents |
|---|---|---|---|
| en | story | medium | 150 |
| en | essay | medium | 150 |
| hi | story | hindisamay | 150 |
| hi | essay | hindisamay | 75 |
| hi | essay | samalochan | 75 |

Two samalochan documents were skipped and replaced (one failed every token cutoff, one the script check).
27 English documents give the same passage in both experiments (one set of rows, `experiments` both), so
English has 573 passages and Hindi 600.

Rows per level (sw / st = rows serving same_words / same_tokens; a row serving both counts in each):

| level | en (sw / st) | hi (sw / st) |
|---|---|---|
| L1_proofread, L2_polish, L3_paraphrase, L4_rewrite (each) | 300 / 300 (573 rows) | 300 / 300 (600 rows) |
| C_continue k75, k50, k25 (each) | 300 / 300 (573 rows) | 300 / 299 |
| OX_extract_outline | 300 / 300 (573 rows) | 300 / 300 |
| G_title | 280 / 280 (536 rows) | 269 / 269 |

Served-cutoff coverage (share of passages whose human anchor exists at N; full table in
`edit_prompts_summary.md`). English documents are shorter (Medium median about 1,000 words), so fewer
reach the large cutoffs; that is expected:

| language / experiment | 100 w | 200 w | 400 w | 800 w | 1200 w | 200 t | 400 t | 800 t | 1500 t |
|---|---|---|---|---|---|---|---|---|---|
| en same_words | 96 % | 94 % | 95 % | 65 % | 43 % | 93 % | 95 % | 65 % | 34 % |
| en same_tokens | 96 % | 93 % | 80 % | 41 % | 35 % | 93 % | 96 % | 82 % | 43 % |
| hi same_words | 92 % | 94 % | 94 % | 96 % | 95 % | 91 % | 93 % | 88 % | 93 % |
| hi same_tokens | 91 % | 90 % | 0 % | 0 % | 0 % | 91 % | 93 % | 88 % | 93 % |

Input cutoffs (the largest each document supports): English same_words passages are cut at 1200 words for
128 documents (27 of them also serve same_tokens at 1500 tokens: the two cuts coincide), 800 for 72, 400
for 97 and 200 for 3; the other 273 same_tokens passages at 1500 tokens for 101, 800 for 123, 400 for 47 and
200 for 2. Hindi: same_words 284 at 1200 words, 14 at 800, 1 each at 400 and 200; same_tokens 279 at 1500
tokens, 18 at 800, 3 at 400.

`max_new_tokens`: en sum 6.09 M (mean 1,188, max 2,560), hi sum 19.63 M (mean 3,679, max 9,472), total
25.7 M: an upper bound, most rows stop at EOS well before it. The OX rows (573 en at 512, 600 hi at 1,024)
run unwatermarked.

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
* Hindi same_words rows are long: about 5,500 input tokens and `max_new_tokens` up to 9,472 for
  L1-L4.
* **English and Hindi passages differ in length.** The genre mix is matched, the lengths are not: Medium
  texts are much shorter than HindiSamay / Samalochan, so at 800 words and above English has fewer human
  anchors (table above). Compare languages at a cutoff only over passages that serve it (`cutoffs`).
* Genre labels are coarse and partly heuristic: Medium's come from the authors' tags, samalochan's from
  the rule above (its essay class can hold a few stories), and HindiSamay's 6 `लघुकथाएँ` count as essays.
* Some samalochan essays are criticism that quotes a lot of verse (about 80 posts have more short lines
  than long ones). They stay in the essay pool, as in earlier builds.
* The caveat numbers above (slack, fall-backs) were measured on the earlier samalochan + hindi-aesthetics
  build; the current coverage is in the table above.
* `edit_prompts.jsonl` is about 156 MB, because each passage appears in `human_passage` and in the prompt.
  Keep it (and `human_passages.jsonl`) out of git: they embed the corpus texts.
