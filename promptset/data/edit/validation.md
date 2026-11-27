# Prompt-set validation

Files: `edit_prompts.jsonl`

Rows: 10455 (languages {'en': 5120, 'hi': 5335}, levels {'L1_proofread': 1173, 'L2_polish': 1173, 'L3_paraphrase': 1173, 'L4_rewrite': 1173, 'C_continue': 3516, 'G_title': 1074, 'OX_extract_outline': 1173}). **0 errors, 7 warnings.**

## Problems

| severity | check | count | description | examples (up to 5) |
|---|---|---|---|---|
| warning | `balance.en_hi_edit` | 7 | edit levels: en and hi row counts differ in a level x split x genre x experiment cell (a row counts in every experiment of its `experiments`) | C_continue_k25/main/story/same_tokens (en=150 hi=149); C_continue_k50/main/story/same_tokens (en=150 hi=149); C_continue_k75/main/story/same_tokens (en=150 hi=149); G_title/main/essay/same_tokens (en=137 hi=119); G_title/main/essay/same_words (en=137 hi=119) |

## Balance

### Rows per level, split, genre and tier (en vs hi)

| level | split | genre | tier | en | hi | match |
|---|---|---|---|---|---|---|
| C_continue_k25 | main | essay | same_tokens | 150 | 150 | ok |
| C_continue_k25 | main | essay | same_words | 150 | 150 | ok |
| C_continue_k25 | main | story | same_tokens | 150 | 149 | MISMATCH |
| C_continue_k25 | main | story | same_words | 150 | 150 | ok |
| C_continue_k50 | main | essay | same_tokens | 150 | 150 | ok |
| C_continue_k50 | main | essay | same_words | 150 | 150 | ok |
| C_continue_k50 | main | story | same_tokens | 150 | 149 | MISMATCH |
| C_continue_k50 | main | story | same_words | 150 | 150 | ok |
| C_continue_k75 | main | essay | same_tokens | 150 | 150 | ok |
| C_continue_k75 | main | essay | same_words | 150 | 150 | ok |
| C_continue_k75 | main | story | same_tokens | 150 | 149 | MISMATCH |
| C_continue_k75 | main | story | same_words | 150 | 150 | ok |
| G_title | main | essay | same_tokens | 137 | 119 | MISMATCH |
| G_title | main | essay | same_words | 137 | 119 | MISMATCH |
| G_title | main | story | same_tokens | 143 | 150 | MISMATCH |
| G_title | main | story | same_words | 143 | 150 | MISMATCH |
| L1_proofread | main | essay | same_tokens | 150 | 150 | ok |
| L1_proofread | main | essay | same_words | 150 | 150 | ok |
| L1_proofread | main | story | same_tokens | 150 | 150 | ok |
| L1_proofread | main | story | same_words | 150 | 150 | ok |
| L2_polish | main | essay | same_tokens | 150 | 150 | ok |
| L2_polish | main | essay | same_words | 150 | 150 | ok |
| L2_polish | main | story | same_tokens | 150 | 150 | ok |
| L2_polish | main | story | same_words | 150 | 150 | ok |
| L3_paraphrase | main | essay | same_tokens | 150 | 150 | ok |
| L3_paraphrase | main | essay | same_words | 150 | 150 | ok |
| L3_paraphrase | main | story | same_tokens | 150 | 150 | ok |
| L3_paraphrase | main | story | same_words | 150 | 150 | ok |
| L4_rewrite | main | essay | same_tokens | 150 | 150 | ok |
| L4_rewrite | main | essay | same_words | 150 | 150 | ok |
| L4_rewrite | main | story | same_tokens | 150 | 150 | ok |
| L4_rewrite | main | story | same_words | 150 | 150 | ok |
| OX_extract_outline | main | essay | same_tokens | 150 | 150 | ok |
| OX_extract_outline | main | essay | same_words | 150 | 150 | ok |
| OX_extract_outline | main | story | same_tokens | 150 | 150 | ok |
| OX_extract_outline | main | story | same_words | 150 | 150 | ok |

## Lengths

### max_new_tokens

| level | language | tier | rows | min | max | sum |
|---|---|---|---|---|---|---|
| C_continue | en | - | 1719 | 256 | 1792 | 1332224 |
| C_continue | hi | - | 1797 | 256 | 6144 | 4176640 |
| G_title | en | - | 536 | 256 | 2304 | 763904 |
| G_title | hi | - | 538 | 512 | 8192 | 2424832 |
| L1_proofread | en | - | 573 | 256 | 2560 | 923904 |
| L1_proofread | hi | - | 600 | 512 | 9472 | 3102976 |
| L2_polish | en | - | 573 | 256 | 2560 | 923904 |
| L2_polish | hi | - | 600 | 512 | 9472 | 3102976 |
| L3_paraphrase | en | - | 573 | 256 | 2560 | 923904 |
| L3_paraphrase | hi | - | 600 | 512 | 9472 | 3102976 |
| L4_rewrite | en | - | 573 | 256 | 2560 | 923904 |
| L4_rewrite | hi | - | 600 | 512 | 9472 | 3102976 |
| OX_extract_outline | en | - | 573 | 512 | 512 | 293376 |
| OX_extract_outline | hi | - | 600 | 1024 | 1024 | 614400 |

### Rendered prompt tokens (apply_chat_template, enable_thinking=False)

| level | language | rows | min | median | max |
|---|---|---|---|---|---|
| C_continue | en | 1719 | 88 | 448 | 1325 |
| C_continue | hi | 1797 | 314 | 1362 | 4935 |
| G_title | en | 536 | 52 | 60 | 74 |
| G_title | hi | 538 | 166 | 182 | 234 |
| L1_proofread | en | 573 | 226 | 1007 | 1698 |
| L1_proofread | hi | 600 | 466 | 1742 | 6420 |
| L2_polish | en | 573 | 223 | 1006 | 1699 |
| L2_polish | hi | 600 | 462 | 1721 | 6415 |
| L3_paraphrase | en | 573 | 219 | 1008 | 1699 |
| L3_paraphrase | hi | 600 | 476 | 1713 | 6390 |
| L4_rewrite | en | 573 | 221 | 1007 | 1701 |
| L4_rewrite | hi | 600 | 475 | 1728 | 6372 |
| OX_extract_outline | en | 573 | 277 | 1060 | 1757 |
| OX_extract_outline | hi | 600 | 667 | 1889 | 6581 |

End of a rendered prompt (what Qwen3 inserts with thinking off is the empty block `<think>\n\n</think>\n\n` after the assistant header):

```
'ut damaging any of the notes.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```
