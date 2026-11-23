# Edit-level prompt set

4597 prompt rows, 600 human passages. Token counts: Qwen3 tokenizer (/mnt/project-files/promptset_stage/qwen3_tokenizer/tokenizer.json).

Length rule: accept iff n >= min(0.95 * N, N - slack), N = cutoff, n = units of the passage cut at the last sentence end at or before N; slack: words 30; tokens en 38, hi 137 = round(30 x tokens_per_word); other languages 0.

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
| G_title | same_tokens | hindi-aesthetics | hi | 123 |
| G_title | same_tokens | samalochan | hi | 77 |
| G_title | same_words | hindi-aesthetics | hi | 123 |
| G_title | same_words | samalochan | hi | 77 |
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
| same_tokens | hindi-aesthetics | hi | tokens | 1500 | 143 |
| same_tokens | hindi-aesthetics | hi | tokens | 400 | 1 |
| same_tokens | hindi-aesthetics | hi | tokens | 800 | 6 |
| same_tokens | samalochan | hi | tokens | 1500 | 129 |
| same_tokens | samalochan | hi | tokens | 200 | 1 |
| same_tokens | samalochan | hi | tokens | 400 | 4 |
| same_tokens | samalochan | hi | tokens | 800 | 16 |
| same_words | hindi-aesthetics | hi | words | 1200 | 121 |
| same_words | hindi-aesthetics | hi | words | 400 | 5 |
| same_words | hindi-aesthetics | hi | words | 800 | 24 |
| same_words | samalochan | hi | words | 1200 | 145 |
| same_words | samalochan | hi | words | 200 | 1 |
| same_words | samalochan | hi | words | 800 | 4 |

## Passages serving each cutoff (human anchor exists at that cutoff)

| experiment | source | lang | kind | N | passages |
|---|---|---|---|---|---|
| same_tokens | hindi-aesthetics | hi | tokens | 1500 | 143 |
| same_tokens | hindi-aesthetics | hi | tokens | 200 | 145 |
| same_tokens | hindi-aesthetics | hi | tokens | 400 | 143 |
| same_tokens | hindi-aesthetics | hi | tokens | 800 | 144 |
| same_tokens | hindi-aesthetics | hi | words | 100 | 144 |
| same_tokens | hindi-aesthetics | hi | words | 200 | 139 |
| same_tokens | samalochan | hi | tokens | 1500 | 129 |
| same_tokens | samalochan | hi | tokens | 200 | 131 |
| same_tokens | samalochan | hi | tokens | 400 | 130 |
| same_tokens | samalochan | hi | tokens | 800 | 123 |
| same_tokens | samalochan | hi | words | 100 | 126 |
| same_tokens | samalochan | hi | words | 200 | 119 |
| same_words | hindi-aesthetics | hi | tokens | 1500 | 143 |
| same_words | hindi-aesthetics | hi | tokens | 200 | 145 |
| same_words | hindi-aesthetics | hi | tokens | 400 | 143 |
| same_words | hindi-aesthetics | hi | tokens | 800 | 144 |
| same_words | hindi-aesthetics | hi | words | 100 | 144 |
| same_words | hindi-aesthetics | hi | words | 1200 | 121 |
| same_words | hindi-aesthetics | hi | words | 200 | 143 |
| same_words | hindi-aesthetics | hi | words | 400 | 142 |
| same_words | hindi-aesthetics | hi | words | 800 | 143 |
| same_words | samalochan | hi | tokens | 1500 | 129 |
| same_words | samalochan | hi | tokens | 200 | 131 |
| same_words | samalochan | hi | tokens | 400 | 130 |
| same_words | samalochan | hi | tokens | 800 | 122 |
| same_words | samalochan | hi | words | 100 | 128 |
| same_words | samalochan | hi | words | 1200 | 145 |
| same_words | samalochan | hi | words | 200 | 134 |
| same_words | samalochan | hi | words | 400 | 137 |
| same_words | samalochan | hi | words | 800 | 139 |

## Upper bound on generated tokens (sum of max_new_tokens)

| level | lang | tokens |
|---|---|---|
| C_continue | hi | 3,923,200 |
| G_title | hi | 1,713,536 |
| L1_proofread | hi | 1,544,896 |
| L2_polish | hi | 1,544,896 |
| L3_paraphrase | hi | 2,996,608 |
| L4_rewrite | hi | 2,996,608 |
| OX_extract_outline | hi | 614,400 |

## Skipped

| level | source | reason | n |
|---|---|---|---|
| (doc) | samalochan | no passage: same_tokens; same_words | 1 |
| C_continue | samalochan | no inner sentence end | 3 |
| G_title | hindi-aesthetics | no usable title | 54 |
| G_title | samalochan | no usable title | 146 |
| L1_proofread | hindi-aesthetics | punctuation_removed: excluded | 300 |
| L2_polish | hindi-aesthetics | punctuation_removed: excluded | 300 |
