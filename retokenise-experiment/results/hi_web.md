# Invisible characters and retokenization drift in web-crawled Hindi

Qwen3 tokenizer (`Qwen/Qwen3-0.6B`, built-in NFC). Web samples: first 1,500 documents streamed from
`allenai/c4` config `hi` (mC4, Common Crawl up to 2020; kept the 775 docs whose letters are at least 80% Devanagari,
the rest are mostly Malayalam/Latin noise) and from `ai4bharat/sangraha` `verified/hin`. Each web document
contributes its first ~200 tokens as one passage. News rows are the earlier 347-passage IITB run.

## How often the characters occur (whole fetched text)

| corpus | docs | ZWJ / 100k chars | ZWNJ | ZWSP | BOM | docs with ZWJ | docs with any of these |
|---|---|---|---|---|---|---|---|
| IITB news (hi) | 3027 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% |
| mC4 web (hi, Devanagari docs) | 775 | 48.7 | 3.1 | 4.7 | 1.3 | 13.0% | 19.6% |
| Sangraha web (hi) | 1500 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% |
| IITB news (en) | 3027 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0% | 0.0% |

## Drift

| corpus | passages | perturbation | passages changed | exact match | 5-gram survival (all) | 5-gram survival (changed only) |
|---|---|---|---|---|---|---|
| IITB news (hi) | 347 | strip_zw | 0 (0.0%) | 1.000 | 1.000 | – |
| mC4 web (hi, Devanagari docs) | 756 | strip_zw | 58 (7.7%) | 0.923 | 0.995 | 0.934 |
| Sangraha web (hi) | 1492 | strip_zw | 0 (0.0%) | 1.000 | 1.000 | – |
| IITB news (en) | 347 | strip_zw | 0 (0.0%) | 1.000 | 1.000 | – |
| IITB news (hi) | 347 | NFKC | 0 (0.0%) | 1.000 | 1.000 | – |
| mC4 web (hi, Devanagari docs) | 756 | NFKC | 11 (1.5%) | 0.985 | 0.999 | 0.956 |
| Sangraha web (hi) | 1492 | NFKC | 2 (0.1%) | 0.999 | 1.000 | 0.961 |
| IITB news (en) | 347 | NFKC | 2 (0.6%) | 0.994 | 1.000 | 0.939 |
| IITB news (hi) | 347 | whitespace | 69 (19.9%) | 0.801 | 0.991 | 0.954 |
| mC4 web (hi, Devanagari docs) | 756 | whitespace | 1 (0.1%) | 0.999 | 1.000 | 0.938 |
| Sangraha web (hi) | 1492 | whitespace | 253 (17.0%) | 0.830 | 0.995 | 0.968 |
| IITB news (en) | 347 | whitespace | 8 (2.3%) | 0.977 | 0.999 | 0.965 |
| IITB news (hi) | 347 | webform | 69 (19.9%) | 0.801 | 0.991 | 0.954 |
| mC4 web (hi, Devanagari docs) | 756 | webform | 68 (9.0%) | 0.910 | 0.994 | 0.934 |
| Sangraha web (hi) | 1492 | webform | 255 (17.1%) | 0.829 | 0.995 | 0.968 |
| IITB news (en) | 347 | webform | 8 (2.3%) | 0.977 | 0.999 | 0.965 |

## Top strip_zw mismatches in mC4

| count | text | code points | original tokens | re-encoded tokens |
|---|---|---|---|---|
| 20 | `﻿` | U+FEFF | `﻿` | `` |
| 17 | `्‍य` | U+094D U+200D U+092F | `् · �#378 · �#235 · य` | `्�#30484 · �#107` |
| 11 | `्‍त` | U+094D U+200D U+0924 | `् · �#378 · �#235 · त` | `्�#30484 · �#97` |
| 8 | `्‍ल` | U+094D U+200D U+0932 | `् · �#378 · �#235 · ल` | `्�#30484 · �#110` |
| 8 | `्‍म` | U+094D U+200D U+092E | `् · �#378 · �#235 · म` | `्�#30484 · �#106` |
| 5 | `्‍व` | U+094D U+200D U+0935 | `् · �#378 · �#235 · व` | `्�#30484 · �#113` |
| 4 | `﻿﻿` | U+FEFF U+FEFF | `﻿ · ﻿` | `` |
| 4 | `्‍स` | U+094D U+200D U+0938 | `् · �#378 · �#235 · स` | `्�#30484 · �#116` |
