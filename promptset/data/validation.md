# Prompt-set validation

Files: `G_prompts.jsonl`

Rows: 602 (languages {'en': 301, 'hi': 301}, levels {'G': 602}). **0 errors, 72 warnings.**

## Problems

| severity | check | count | description | examples (up to 5) |
|---|---|---|---|---|
| warning | `balance.config` | 11 | counts differ from config/promptset.yaml (per_genre, pilot_per_genre_tier) | main+pilot/story (92 != 100); reserve/story (48 != 50); main+pilot/opinion (94 != 100); reserve/opinion (1 != 50); pilot/opinion/short (2 != 3) |
| warning | `balance.hindi_share` | 2 | Hindi-native share of a genre is more than 0.1 off hindi_native_share | opinion (0.03 vs 0.5); personal (0.03 vs 0.5) |
| warning | `content.length_cap_soft` | 14 | translated prompt_text matches a non-numeric length-cap pattern (often a false positive, e.g. 'छोटा सा') | G-essay-a2637c2a59-hi (एक पंक्ति); G-story-70edaeae46-hi (छोटी सी); G-story-cccce585c1-hi (छोटी सी); G-story-6904557fdd-hi (छोटी सी); G-story-7efaf70128-hi (छोटी सी) |
| warning | `content.needs_input` | 33 | prompt_text matches a needs-input pattern (below, the following, image, ...) | G-story-026b6ecac5-en (the story); G-story-3e7f3d9d93-en (The story); G-opinion-87827c7221-en (continue); G-opinion-45f82881e2-en (video); G-opinion-45f82881e2-hi (वीडियो) |
| warning | `content.short_form_soft` | 10 | translated prompt_text matches a short-form pattern (check: 'शेर' = lion, 'table', ...) | G-story-8bec5b9bcf-en (script); G-opinion-87827c7221-hi (अवकाश); G-opinion-7a3cb16546-hi (गीत); G-personal-5a07868076-hi (गीत); G-story-94823aa887-en (script) |
| warning | `tr.latin_leftover` | 2 | Hindi translation keeps lowercase Latin words (untranslated leftovers; capitalised words are taken as names) | G-opinion-71b93d5ad3-hi (k); G-story-04a46463f8-hi (m m) |

## Balance

### Rows per level, split, genre and tier (en vs hi)

| level | split | genre | tier | en | hi | match |
|---|---|---|---|---|---|---|
| G | main | essay | long | 11 | 11 | ok |
| G | main | essay | short | 12 | 12 | ok |
| G | main | opinion | long | 44 | 44 | ok |
| G | main | opinion | short | 45 | 45 | ok |
| G | main | personal | long | 6 | 6 | ok |
| G | main | personal | short | 9 | 9 | ok |
| G | main | story | long | 44 | 44 | ok |
| G | main | story | short | 42 | 42 | ok |
| G | pilot | essay | long | 3 | 3 | ok |
| G | pilot | essay | short | 3 | 3 | ok |
| G | pilot | opinion | long | 3 | 3 | ok |
| G | pilot | opinion | short | 2 | 2 | ok |
| G | pilot | personal | short | 1 | 1 | ok |
| G | pilot | story | long | 3 | 3 | ok |
| G | pilot | story | short | 3 | 3 | ok |
| G | reserve | essay | long | 3 | 3 | ok |
| G | reserve | essay | short | 3 | 3 | ok |
| G | reserve | opinion | short | 1 | 1 | ok |
| G | reserve | personal | long | 8 | 8 | ok |
| G | reserve | personal | short | 7 | 7 | ok |
| G | reserve | story | long | 25 | 25 | ok |
| G | reserve | story | short | 23 | 23 | ok |

### Hindi-native share per genre (G, 301 pairs)

| genre | pairs | hi-native | en-native | share | target |
|---|---|---|---|---|---|
| essay | 35 | 20 | 15 | 0.57 | 0.5 |
| opinion | 95 | 3 | 92 | 0.03 | 0.5 |
| personal | 31 | 1 | 30 | 0.03 | 0.5 |
| story | 140 | 69 | 71 | 0.49 | 0.5 |

### Native language per split and genre (pairs)

| split | genre | en-native | hi-native |
|---|---|---|---|
| pilot | essay | 2 | 4 |
| pilot | opinion | 2 | 3 |
| pilot | personal | 0 | 1 |
| pilot | story | 2 | 4 |
| main | essay | 13 | 10 |
| main | opinion | 89 | 0 |
| main | personal | 15 | 0 |
| main | story | 45 | 41 |
| reserve | essay | 0 | 6 |
| reserve | opinion | 1 | 0 |
| reserve | personal | 15 | 0 |
| reserve | story | 24 | 24 |

### Source mix per genre (pairs)

| genre | source | native | pairs |
|---|---|---|---|
| essay | anudesh | en | 1 |
| essay | anudesh | hi | 11 |
| essay | aya | hi | 9 |
| essay | no_robots | en | 3 |
| essay | oasst2 | en | 3 |
| essay | persuade | en | 1 |
| essay | wildchat | en | 7 |
| opinion | anudesh | hi | 1 |
| opinion | aya_eval_dolly | hi | 2 |
| opinion | dolly | en | 1 |
| opinion | no_robots | en | 11 |
| opinion | oasst2 | en | 1 |
| opinion | persuade | en | 7 |
| opinion | wildchat | en | 72 |
| personal | aya | hi | 1 |
| personal | dolly | en | 1 |
| personal | no_robots | en | 12 |
| personal | wildchat | en | 17 |
| story | anudesh | en | 4 |
| story | anudesh | hi | 31 |
| story | aya | hi | 35 |
| story | aya_eval_dolly | hi | 3 |
| story | dolly | en | 6 |
| story | no_robots | en | 13 |
| story | oasst2 | en | 6 |
| story | wildchat | en | 11 |
| story | writingprompts | en | 31 |

## Lengths

### max_new_tokens

| level | language | tier | rows | min | max | sum |
|---|---|---|---|---|---|---|
| G | en | long | 150 | 2176 | 2176 | 326400 |
| G | en | short | 151 | 704 | 704 | 106304 |
| G | hi | long | 150 | 7744 | 7744 | 1161600 |
| G | hi | short | 151 | 2432 | 2432 | 367232 |

### Rendered prompt tokens (apply_chat_template, enable_thinking=False)

| level | language | rows | min | median | max |
|---|---|---|---|---|---|
| G | en | 301 | 55 | 70 | 191 |
| G | hi | 301 | 174 | 252 | 808 |

End of a rendered prompt (what Qwen3 inserts with thinking off is the empty block `<think>\n\n</think>\n\n` after the assistant header):

```
'logy. Write about 1300 words.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

## Translation sanity

Words in translated prompt_text / words in original_text:

| direction | rows | min | median | max |
|---|---|---|---|---|
| en->hi | 208 | 0.84 | 1.18 | 1.74 |
| hi->en | 93 | 0.63 | 0.87 | 1.24 |

Hindi translations with lowercase Latin words: 2 rows (3 words); capitalised Latin words (taken as names): 0.
