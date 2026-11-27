# Edit-level prompt set

10455 prompt rows, 1173 human passages. Token counts: Qwen3 tokenizer (/mnt/project-files/promptset_stage/qwen3_tokenizer/tokenizer.json).

Length rule: accept iff n >= min(0.95 * N, N - slack), N = cutoff, n = units of the passage cut at the last sentence end at or before N; slack: words 30; tokens en 38, hi 137 = round(30 x tokens_per_word); other languages 0.

Passage script check (trim.py's threshold): a document is skipped and replaced by the next one in the sample if a passage has less than 0.9 Latin (en), 0.8 Devanagari (hi) among its letters; 1 skipped.

max_new_tokens = ceil256(max(level factor x requested tokens, 1.25 x (largest served token cutoff - head tokens), largest served word cutoff x passage tokens/word - head tokens)); prompt + max_new_tokens <= 32768.

## Documents (genre-matched sample)

300 documents per language, genre mix {'story': 0.5, 'essay': 0.5}; quota per genre and source from sampling.genre_sources (a source short of usable documents is topped up by the other sources of its genre). Genres outside the mix are not sampled (quota 0).

| lang/genre/source | available | quota | sampled |
|---|---|---|---|
| en/essay/medium | 2849 | 150 | 150 |
| en/story/medium | 1143 | 150 | 150 |
| hi/essay/hindisamay | 661 | 75 | 75 |
| hi/essay/samalochan | 540 | 75 | 75 |
| hi/story/hindisamay | 1035 | 150 | 150 |
| hi/story/samalochan | 36 | 0 | 0 |

## Sampled documents per language x genre x source

| lang | genre | source | docs |
|---|---|---|---|
| en | essay | medium | 150 |
| en | story | medium | 150 |
| hi | essay | hindisamay | 75 |
| hi | essay | samalochan | 75 |
| hi | story | hindisamay | 150 |

## Rows serving each experiment, per level x language x genre

A row whose words and tokens cuts coincide serves both experiments and is counted under each (filter on `experiments`).

| level | lang | genre | experiment | rows |
|---|---|---|---|---|
| C_continue_k25 | en | essay | same_tokens | 150 |
| C_continue_k25 | en | essay | same_words | 150 |
| C_continue_k25 | en | story | same_tokens | 150 |
| C_continue_k25 | en | story | same_words | 150 |
| C_continue_k25 | hi | essay | same_tokens | 150 |
| C_continue_k25 | hi | essay | same_words | 150 |
| C_continue_k25 | hi | story | same_tokens | 149 |
| C_continue_k25 | hi | story | same_words | 150 |
| C_continue_k50 | en | essay | same_tokens | 150 |
| C_continue_k50 | en | essay | same_words | 150 |
| C_continue_k50 | en | story | same_tokens | 150 |
| C_continue_k50 | en | story | same_words | 150 |
| C_continue_k50 | hi | essay | same_tokens | 150 |
| C_continue_k50 | hi | essay | same_words | 150 |
| C_continue_k50 | hi | story | same_tokens | 149 |
| C_continue_k50 | hi | story | same_words | 150 |
| C_continue_k75 | en | essay | same_tokens | 150 |
| C_continue_k75 | en | essay | same_words | 150 |
| C_continue_k75 | en | story | same_tokens | 150 |
| C_continue_k75 | en | story | same_words | 150 |
| C_continue_k75 | hi | essay | same_tokens | 150 |
| C_continue_k75 | hi | essay | same_words | 150 |
| C_continue_k75 | hi | story | same_tokens | 149 |
| C_continue_k75 | hi | story | same_words | 150 |
| G_title | en | essay | same_tokens | 137 |
| G_title | en | essay | same_words | 137 |
| G_title | en | story | same_tokens | 143 |
| G_title | en | story | same_words | 143 |
| G_title | hi | essay | same_tokens | 119 |
| G_title | hi | essay | same_words | 119 |
| G_title | hi | story | same_tokens | 150 |
| G_title | hi | story | same_words | 150 |
| L1_proofread | en | essay | same_tokens | 150 |
| L1_proofread | en | essay | same_words | 150 |
| L1_proofread | en | story | same_tokens | 150 |
| L1_proofread | en | story | same_words | 150 |
| L1_proofread | hi | essay | same_tokens | 150 |
| L1_proofread | hi | essay | same_words | 150 |
| L1_proofread | hi | story | same_tokens | 150 |
| L1_proofread | hi | story | same_words | 150 |
| L2_polish | en | essay | same_tokens | 150 |
| L2_polish | en | essay | same_words | 150 |
| L2_polish | en | story | same_tokens | 150 |
| L2_polish | en | story | same_words | 150 |
| L2_polish | hi | essay | same_tokens | 150 |
| L2_polish | hi | essay | same_words | 150 |
| L2_polish | hi | story | same_tokens | 150 |
| L2_polish | hi | story | same_words | 150 |
| L3_paraphrase | en | essay | same_tokens | 150 |
| L3_paraphrase | en | essay | same_words | 150 |
| L3_paraphrase | en | story | same_tokens | 150 |
| L3_paraphrase | en | story | same_words | 150 |
| L3_paraphrase | hi | essay | same_tokens | 150 |
| L3_paraphrase | hi | essay | same_words | 150 |
| L3_paraphrase | hi | story | same_tokens | 150 |
| L3_paraphrase | hi | story | same_words | 150 |
| L4_rewrite | en | essay | same_tokens | 150 |
| L4_rewrite | en | essay | same_words | 150 |
| L4_rewrite | en | story | same_tokens | 150 |
| L4_rewrite | en | story | same_words | 150 |
| L4_rewrite | hi | essay | same_tokens | 150 |
| L4_rewrite | hi | essay | same_words | 150 |
| L4_rewrite | hi | story | same_tokens | 150 |
| L4_rewrite | hi | story | same_words | 150 |
| OX_extract_outline | en | essay | same_tokens | 150 |
| OX_extract_outline | en | essay | same_words | 150 |
| OX_extract_outline | en | story | same_tokens | 150 |
| OX_extract_outline | en | story | same_words | 150 |
| OX_extract_outline | hi | essay | same_tokens | 150 |
| OX_extract_outline | hi | essay | same_words | 150 |
| OX_extract_outline | hi | story | same_tokens | 150 |
| OX_extract_outline | hi | story | same_words | 150 |

## Rows per level x experiment x source x language

| level | experiment | source | lang | rows |
|---|---|---|---|---|
| C_continue_k25 | same_tokens | hindisamay | hi | 224 |
| C_continue_k25 | same_tokens | medium | en | 273 |
| C_continue_k25 | same_tokens | samalochan | hi | 75 |
| C_continue_k25 | same_words | hindisamay | hi | 225 |
| C_continue_k25 | same_words | medium | en | 273 |
| C_continue_k25 | same_words | samalochan | hi | 75 |
| C_continue_k25 | same_words+same_tokens | medium | en | 27 |
| C_continue_k50 | same_tokens | hindisamay | hi | 224 |
| C_continue_k50 | same_tokens | medium | en | 273 |
| C_continue_k50 | same_tokens | samalochan | hi | 75 |
| C_continue_k50 | same_words | hindisamay | hi | 225 |
| C_continue_k50 | same_words | medium | en | 273 |
| C_continue_k50 | same_words | samalochan | hi | 75 |
| C_continue_k50 | same_words+same_tokens | medium | en | 27 |
| C_continue_k75 | same_tokens | hindisamay | hi | 224 |
| C_continue_k75 | same_tokens | medium | en | 273 |
| C_continue_k75 | same_tokens | samalochan | hi | 75 |
| C_continue_k75 | same_words | hindisamay | hi | 225 |
| C_continue_k75 | same_words | medium | en | 273 |
| C_continue_k75 | same_words | samalochan | hi | 75 |
| C_continue_k75 | same_words+same_tokens | medium | en | 27 |
| G_title | same_tokens | hindisamay | hi | 225 |
| G_title | same_tokens | medium | en | 256 |
| G_title | same_tokens | samalochan | hi | 44 |
| G_title | same_words | hindisamay | hi | 225 |
| G_title | same_words | medium | en | 256 |
| G_title | same_words | samalochan | hi | 44 |
| G_title | same_words+same_tokens | medium | en | 24 |
| L1_proofread | same_tokens | hindisamay | hi | 225 |
| L1_proofread | same_tokens | medium | en | 273 |
| L1_proofread | same_tokens | samalochan | hi | 75 |
| L1_proofread | same_words | hindisamay | hi | 225 |
| L1_proofread | same_words | medium | en | 273 |
| L1_proofread | same_words | samalochan | hi | 75 |
| L1_proofread | same_words+same_tokens | medium | en | 27 |
| L2_polish | same_tokens | hindisamay | hi | 225 |
| L2_polish | same_tokens | medium | en | 273 |
| L2_polish | same_tokens | samalochan | hi | 75 |
| L2_polish | same_words | hindisamay | hi | 225 |
| L2_polish | same_words | medium | en | 273 |
| L2_polish | same_words | samalochan | hi | 75 |
| L2_polish | same_words+same_tokens | medium | en | 27 |
| L3_paraphrase | same_tokens | hindisamay | hi | 225 |
| L3_paraphrase | same_tokens | medium | en | 273 |
| L3_paraphrase | same_tokens | samalochan | hi | 75 |
| L3_paraphrase | same_words | hindisamay | hi | 225 |
| L3_paraphrase | same_words | medium | en | 273 |
| L3_paraphrase | same_words | samalochan | hi | 75 |
| L3_paraphrase | same_words+same_tokens | medium | en | 27 |
| L4_rewrite | same_tokens | hindisamay | hi | 225 |
| L4_rewrite | same_tokens | medium | en | 273 |
| L4_rewrite | same_tokens | samalochan | hi | 75 |
| L4_rewrite | same_words | hindisamay | hi | 225 |
| L4_rewrite | same_words | medium | en | 273 |
| L4_rewrite | same_words | samalochan | hi | 75 |
| L4_rewrite | same_words+same_tokens | medium | en | 27 |
| OX_extract_outline | same_tokens | hindisamay | hi | 225 |
| OX_extract_outline | same_tokens | medium | en | 273 |
| OX_extract_outline | same_tokens | samalochan | hi | 75 |
| OX_extract_outline | same_words | hindisamay | hi | 225 |
| OX_extract_outline | same_words | medium | en | 273 |
| OX_extract_outline | same_words | samalochan | hi | 75 |
| OX_extract_outline | same_words+same_tokens | medium | en | 27 |

## Passages per experiment and input cutoff

| experiment | source | lang | cut by | cutoff | passages |
|---|---|---|---|---|---|
| same_tokens | hindisamay | hi | tokens | 1500 | 212 |
| same_tokens | hindisamay | hi | tokens | 400 | 3 |
| same_tokens | hindisamay | hi | tokens | 800 | 10 |
| same_tokens | medium | en | tokens | 1500 | 101 |
| same_tokens | medium | en | tokens | 200 | 2 |
| same_tokens | medium | en | tokens | 400 | 47 |
| same_tokens | medium | en | tokens | 800 | 123 |
| same_tokens | samalochan | hi | tokens | 1500 | 67 |
| same_tokens | samalochan | hi | tokens | 800 | 8 |
| same_words | hindisamay | hi | words | 1200 | 212 |
| same_words | hindisamay | hi | words | 200 | 1 |
| same_words | hindisamay | hi | words | 400 | 1 |
| same_words | hindisamay | hi | words | 800 | 11 |
| same_words | medium | en | words | 1200 | 101 |
| same_words | medium | en | words | 200 | 3 |
| same_words | medium | en | words | 400 | 97 |
| same_words | medium | en | words | 800 | 72 |
| same_words | samalochan | hi | words | 1200 | 72 |
| same_words | samalochan | hi | words | 800 | 3 |
| same_words+same_tokens | medium | en | words | 1200 | 27 |

## Passages serving each cutoff (human anchor exists at that cutoff)

| experiment | source | lang | kind | N | passages |
|---|---|---|---|---|---|
| same_tokens | hindisamay | hi | tokens | 1500 | 212 |
| same_tokens | hindisamay | hi | tokens | 200 | 206 |
| same_tokens | hindisamay | hi | tokens | 400 | 213 |
| same_tokens | hindisamay | hi | tokens | 800 | 202 |
| same_tokens | hindisamay | hi | words | 100 | 210 |
| same_tokens | hindisamay | hi | words | 200 | 207 |
| same_tokens | medium | en | tokens | 1500 | 101 |
| same_tokens | medium | en | tokens | 200 | 255 |
| same_tokens | medium | en | tokens | 400 | 266 |
| same_tokens | medium | en | tokens | 800 | 219 |
| same_tokens | medium | en | words | 100 | 262 |
| same_tokens | medium | en | words | 1200 | 78 |
| same_tokens | medium | en | words | 200 | 253 |
| same_tokens | medium | en | words | 400 | 216 |
| same_tokens | medium | en | words | 800 | 97 |
| same_tokens | samalochan | hi | tokens | 1500 | 67 |
| same_tokens | samalochan | hi | tokens | 200 | 67 |
| same_tokens | samalochan | hi | tokens | 400 | 66 |
| same_tokens | samalochan | hi | tokens | 800 | 63 |
| same_tokens | samalochan | hi | words | 100 | 64 |
| same_tokens | samalochan | hi | words | 200 | 62 |
| same_words | hindisamay | hi | tokens | 1500 | 212 |
| same_words | hindisamay | hi | tokens | 200 | 206 |
| same_words | hindisamay | hi | tokens | 400 | 213 |
| same_words | hindisamay | hi | tokens | 800 | 202 |
| same_words | hindisamay | hi | words | 100 | 211 |
| same_words | hindisamay | hi | words | 1200 | 212 |
| same_words | hindisamay | hi | words | 200 | 214 |
| same_words | hindisamay | hi | words | 400 | 213 |
| same_words | hindisamay | hi | words | 800 | 218 |
| same_words | medium | en | tokens | 1500 | 76 |
| same_words | medium | en | tokens | 200 | 255 |
| same_words | medium | en | tokens | 400 | 263 |
| same_words | medium | en | tokens | 800 | 168 |
| same_words | medium | en | words | 100 | 262 |
| same_words | medium | en | words | 1200 | 101 |
| same_words | medium | en | words | 200 | 255 |
| same_words | medium | en | words | 400 | 262 |
| same_words | medium | en | words | 800 | 169 |
| same_words | samalochan | hi | tokens | 1500 | 67 |
| same_words | samalochan | hi | tokens | 200 | 67 |
| same_words | samalochan | hi | tokens | 400 | 66 |
| same_words | samalochan | hi | tokens | 800 | 63 |
| same_words | samalochan | hi | words | 100 | 64 |
| same_words | samalochan | hi | words | 1200 | 72 |
| same_words | samalochan | hi | words | 200 | 68 |
| same_words | samalochan | hi | words | 400 | 68 |
| same_words | samalochan | hi | words | 800 | 70 |
| same_words+same_tokens | medium | en | tokens | 1500 | 27 |
| same_words+same_tokens | medium | en | tokens | 200 | 25 |
| same_words+same_tokens | medium | en | tokens | 400 | 21 |
| same_words+same_tokens | medium | en | tokens | 800 | 27 |
| same_words+same_tokens | medium | en | words | 100 | 25 |
| same_words+same_tokens | medium | en | words | 1200 | 27 |
| same_words+same_tokens | medium | en | words | 200 | 26 |
| same_words+same_tokens | medium | en | words | 400 | 23 |
| same_words+same_tokens | medium | en | words | 800 | 27 |

## Served-cutoff coverage per language

Passages (serving each experiment) whose human anchor exists at cutoff N, out of all passages of that language and experiment.

| lang | experiment | kind | N | passages | of | share |
|---|---|---|---|---|---|---|
| en | same_tokens | words | 100 | 287 | 300 | 96% |
| en | same_tokens | words | 200 | 279 | 300 | 93% |
| en | same_tokens | words | 400 | 239 | 300 | 80% |
| en | same_tokens | words | 800 | 124 | 300 | 41% |
| en | same_tokens | words | 1200 | 105 | 300 | 35% |
| en | same_tokens | tokens | 200 | 280 | 300 | 93% |
| en | same_tokens | tokens | 400 | 287 | 300 | 96% |
| en | same_tokens | tokens | 800 | 246 | 300 | 82% |
| en | same_tokens | tokens | 1500 | 128 | 300 | 43% |
| en | same_words | words | 100 | 287 | 300 | 96% |
| en | same_words | words | 200 | 281 | 300 | 94% |
| en | same_words | words | 400 | 285 | 300 | 95% |
| en | same_words | words | 800 | 196 | 300 | 65% |
| en | same_words | words | 1200 | 128 | 300 | 43% |
| en | same_words | tokens | 200 | 280 | 300 | 93% |
| en | same_words | tokens | 400 | 284 | 300 | 95% |
| en | same_words | tokens | 800 | 195 | 300 | 65% |
| en | same_words | tokens | 1500 | 103 | 300 | 34% |
| hi | same_tokens | words | 100 | 274 | 300 | 91% |
| hi | same_tokens | words | 200 | 269 | 300 | 90% |
| hi | same_tokens | words | 400 | 0 | 300 | 0% |
| hi | same_tokens | words | 800 | 0 | 300 | 0% |
| hi | same_tokens | words | 1200 | 0 | 300 | 0% |
| hi | same_tokens | tokens | 200 | 273 | 300 | 91% |
| hi | same_tokens | tokens | 400 | 279 | 300 | 93% |
| hi | same_tokens | tokens | 800 | 265 | 300 | 88% |
| hi | same_tokens | tokens | 1500 | 279 | 300 | 93% |
| hi | same_words | words | 100 | 275 | 300 | 92% |
| hi | same_words | words | 200 | 282 | 300 | 94% |
| hi | same_words | words | 400 | 281 | 300 | 94% |
| hi | same_words | words | 800 | 288 | 300 | 96% |
| hi | same_words | words | 1200 | 284 | 300 | 95% |
| hi | same_words | tokens | 200 | 273 | 300 | 91% |
| hi | same_words | tokens | 400 | 279 | 300 | 93% |
| hi | same_words | tokens | 800 | 265 | 300 | 88% |
| hi | same_words | tokens | 1500 | 279 | 300 | 93% |

## Upper bound on generated tokens (sum of max_new_tokens)

| level | lang | tokens |
|---|---|---|
| C_continue | en | 1,332,224 |
| C_continue | hi | 4,176,640 |
| G_title | en | 763,904 |
| G_title | hi | 2,424,832 |
| L1_proofread | en | 923,904 |
| L1_proofread | hi | 3,102,976 |
| L2_polish | en | 923,904 |
| L2_polish | hi | 3,102,976 |
| L3_paraphrase | en | 923,904 |
| L3_paraphrase | hi | 3,102,976 |
| L4_rewrite | en | 923,904 |
| L4_rewrite | hi | 3,102,976 |
| OX_extract_outline | en | 293,376 |
| OX_extract_outline | hi | 614,400 |

## max_new_tokens per language and split

| lang | split | rows | mean | max | sum |
|---|---|---|---|---|---|
| en | main | 5120 | 1,188 | 2,560 | 6,085,120 |
| hi | main | 5335 | 3,679 | 9,472 | 19,627,776 |

## Skipped

| level | source | reason | n |
|---|---|---|---|
| (doc) | samalochan | no passage: same_tokens; same_words | 1 |
| (doc) | samalochan | passage script share < 0.8: same_tokens; same_words | 1 |
| C_continue | hindisamay | no inner sentence end | 3 |
| G_title | medium | no usable title | 37 |
| G_title | samalochan | no usable title | 62 |
