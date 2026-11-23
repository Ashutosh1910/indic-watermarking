# Edit-level prompt set

4601 prompt rows, 600 human passages. Token counts: Qwen3 tokenizer (/mnt/project-files/promptset_stage/qwen3_tokenizer/tokenizer.json).

## Documents

| source/lang | available | sampled |
|---|---|---|
| hindi-aesthetics/hi | 968 | 150 |
| samalochan/hi | 576 | 150 |

## Rows per level x experiment x source x language

| level | experiment | source | lang | rows |
|---|---|---|---|---|
| C_continue_k25 | same_tokens | hindi-aesthetics | hi | 150 |
| C_continue_k25 | same_tokens | samalochan | hi | 149 |
| C_continue_k25 | same_words | hindi-aesthetics | hi | 150 |
| C_continue_k25 | same_words | samalochan | hi | 150 |
| C_continue_k50 | same_tokens | hindi-aesthetics | hi | 150 |
| C_continue_k50 | same_tokens | samalochan | hi | 149 |
| C_continue_k50 | same_words | hindi-aesthetics | hi | 150 |
| C_continue_k50 | same_words | samalochan | hi | 150 |
| C_continue_k75 | same_tokens | hindi-aesthetics | hi | 150 |
| C_continue_k75 | same_tokens | samalochan | hi | 149 |
| C_continue_k75 | same_words | hindi-aesthetics | hi | 150 |
| C_continue_k75 | same_words | samalochan | hi | 150 |
| G_title | same_tokens | hindi-aesthetics | hi | 124 |
| G_title | same_tokens | samalochan | hi | 78 |
| G_title | same_words | hindi-aesthetics | hi | 124 |
| G_title | same_words | samalochan | hi | 78 |
| L1_proofread | same_tokens | samalochan | hi | 150 |
| L1_proofread | same_words | samalochan | hi | 150 |
| L2_polish | same_tokens | samalochan | hi | 150 |
| L2_polish | same_words | samalochan | hi | 150 |
| L3_paraphrase | same_tokens | hindi-aesthetics | hi | 150 |
| L3_paraphrase | same_tokens | samalochan | hi | 150 |
| L3_paraphrase | same_words | hindi-aesthetics | hi | 150 |
| L3_paraphrase | same_words | samalochan | hi | 150 |
| L4_rewrite | same_tokens | hindi-aesthetics | hi | 150 |
| L4_rewrite | same_tokens | samalochan | hi | 150 |
| L4_rewrite | same_words | hindi-aesthetics | hi | 150 |
| L4_rewrite | same_words | samalochan | hi | 150 |
| OX_extract_outline | same_tokens | hindi-aesthetics | hi | 150 |
| OX_extract_outline | same_tokens | samalochan | hi | 150 |
| OX_extract_outline | same_words | hindi-aesthetics | hi | 150 |
| OX_extract_outline | same_words | samalochan | hi | 150 |

## Passages per experiment and input cutoff

| experiment | source | lang | cut by | cutoff | passages |
|---|---|---|---|---|---|
| same_tokens | hindi-aesthetics | hi | tokens | 1500 | 133 |
| same_tokens | hindi-aesthetics | hi | tokens | 200 | 1 |
| same_tokens | hindi-aesthetics | hi | tokens | 400 | 5 |
| same_tokens | hindi-aesthetics | hi | tokens | 800 | 11 |
| same_tokens | samalochan | hi | tokens | 1500 | 115 |
| same_tokens | samalochan | hi | tokens | 200 | 3 |
| same_tokens | samalochan | hi | tokens | 400 | 9 |
| same_tokens | samalochan | hi | tokens | 800 | 23 |
| same_words | hindi-aesthetics | hi | words | 100 | 1 |
| same_words | hindi-aesthetics | hi | words | 1200 | 122 |
| same_words | hindi-aesthetics | hi | words | 400 | 4 |
| same_words | hindi-aesthetics | hi | words | 800 | 23 |
| same_words | samalochan | hi | words | 1200 | 146 |
| same_words | samalochan | hi | words | 800 | 4 |

## Passages serving each cutoff (human anchor exists at that cutoff)

| experiment | source | lang | kind | N | passages |
|---|---|---|---|---|---|
| same_tokens | hindi-aesthetics | hi | tokens | 1500 | 133 |
| same_tokens | hindi-aesthetics | hi | tokens | 200 | 34 |
| same_tokens | hindi-aesthetics | hi | tokens | 400 | 56 |
| same_tokens | hindi-aesthetics | hi | tokens | 800 | 90 |
| same_tokens | hindi-aesthetics | hi | words | 100 | 70 |
| same_tokens | hindi-aesthetics | hi | words | 200 | 94 |
| same_tokens | samalochan | hi | tokens | 1500 | 115 |
| same_tokens | samalochan | hi | tokens | 200 | 20 |
| same_tokens | samalochan | hi | tokens | 400 | 37 |
| same_tokens | samalochan | hi | tokens | 800 | 71 |
| same_tokens | samalochan | hi | words | 100 | 33 |
| same_tokens | samalochan | hi | words | 200 | 54 |
| same_words | hindi-aesthetics | hi | tokens | 1500 | 132 |
| same_words | hindi-aesthetics | hi | tokens | 200 | 34 |
| same_words | hindi-aesthetics | hi | tokens | 400 | 56 |
| same_words | hindi-aesthetics | hi | tokens | 800 | 89 |
| same_words | hindi-aesthetics | hi | words | 100 | 71 |
| same_words | hindi-aesthetics | hi | words | 1200 | 122 |
| same_words | hindi-aesthetics | hi | words | 200 | 104 |
| same_words | hindi-aesthetics | hi | words | 400 | 134 |
| same_words | hindi-aesthetics | hi | words | 800 | 143 |
| same_words | samalochan | hi | tokens | 1500 | 115 |
| same_words | samalochan | hi | tokens | 200 | 20 |
| same_words | samalochan | hi | tokens | 400 | 37 |
| same_words | samalochan | hi | tokens | 800 | 71 |
| same_words | samalochan | hi | words | 100 | 35 |
| same_words | samalochan | hi | words | 1200 | 146 |
| same_words | samalochan | hi | words | 200 | 70 |
| same_words | samalochan | hi | words | 400 | 125 |
| same_words | samalochan | hi | words | 800 | 141 |

## Upper bound on generated tokens (sum of max_new_tokens)

| level | lang | tokens |
|---|---|---|
| C_continue | hi | 3,898,176 |
| G_title | hi | 1,726,336 |
| L1_proofread | hi | 1,535,936 |
| L2_polish | hi | 1,535,936 |
| L3_paraphrase | hi | 2,977,472 |
| L4_rewrite | hi | 2,977,472 |
| OX_extract_outline | hi | 614,400 |

## Skipped

| level | source | reason | n |
|---|---|---|---|
| (doc) | hindi-aesthetics | no passage: same_tokens | 8 |
| (doc) | samalochan | no passage: same_tokens | 25 |
| (doc) | samalochan | no passage: same_tokens; same_words | 2 |
| C_continue | samalochan | no inner sentence end | 3 |
| G_title | hindi-aesthetics | no usable title | 52 |
| G_title | samalochan | no usable title | 144 |
| L1_proofread | hindi-aesthetics | punctuation_removed: excluded | 300 |
| L2_polish | hindi-aesthetics | punctuation_removed: excluded | 300 |
