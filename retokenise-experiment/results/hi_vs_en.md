# Retokenization drift: Hindi vs English

| | Hindi | English |
|---|---|---|
| passages | 347 | 347 |
| tokens / passage | 195 | 194 |
| chars / token | 1.09 | 4.72 |
| tokens / word | 4.71 | 1.27 |

Hindi: Text source: `{ "lang": "hi", "dataset": "cfilt/iitb-english-hindi", "config": null, "splits": [ "test", "validation" ], "field": "translation.hi", "n_lines": 3027 }`

English: Text source: `{ "lang": "en", "dataset": "cfilt/iitb-english-hindi", "config": null, "splits": [ "test", "validation" ], "field": "translation.en", "n_lines": 3027 }`

## All passages

Each cell is hi / en.

| perturbation | text changed | exact match | aligned pos. | 5-gram survival | Δ tokens |
|---|---|---|---|---|---|
| none | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.00 / 0.00 |
| NFC | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.00 / 0.00 |
| NFD | 0.006 / 0.012 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.00 / 0.00 |
| NFKC | 0.000 / 0.006 | 1.000 / 0.994 | 1.000 / 1.000 | 1.000 / 1.000 | 0.00 / -0.02 |
| strip_zw | 0.000 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.00 / 0.00 |
| whitespace | 0.199 / 0.023 | 0.801 / 0.977 | 0.998 / 1.000 | 0.991 / 0.999 | -0.35 / -0.03 |
| nukta_precomposed | 0.372 / 0.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 0.00 / 0.00 |
| webform | 0.199 / 0.023 | 0.801 / 0.977 | 0.998 / 1.000 | 0.991 / 0.999 | -0.35 / -0.03 |

## Only passages the perturbation changed (cost per hit)

Each cell is hi / en; `–` means the perturbation never changed a passage in that language.

| perturbation | n changed | chars changed / passage | exact match | aligned pos. | 5-gram survival |
|---|---|---|---|---|---|
| none | 0 / 0 | – / – | – / – | – / – | – / – |
| NFC | 0 / 0 | – / – | – / – | – / – | – / – |
| NFD | 2 / 4 | 3.0 / 3.8 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| NFKC | 0 / 2 | – / 5.0 | – / 0.000 | – / 0.972 | – / 0.939 |
| strip_zw | 0 / 0 | – / – | – / – | – / – | – / – |
| whitespace | 69 / 8 | 1.8 / 2.0 | 0.000 / 0.000 | 0.991 / 0.989 | 0.954 / 0.965 |
| nukta_precomposed | 129 / 0 | 5.1 / – | 1.000 / – | 1.000 / – | 1.000 / – |
| webform | 69 / 8 | 1.8 / 2.0 | 0.000 / 0.000 | 0.991 / 0.989 | 0.954 / 0.965 |

## Top mismatches

Original span (code points), original tokens → re-encoded tokens, how often, and context.

### Hindi

**whitespace**

| count | text | code points | original tokens | re-encoded tokens | context |
|---|---|---|---|---|---|
| 120 | ` ` | U+0020 | `␣` | `` | ार को  मुला� |
| 3 | `  ` | U+0020 U+0020 | `␣␣` | `` | ्यपि   कैर |

**webform**

| count | text | code points | original tokens | re-encoded tokens | context |
|---|---|---|---|---|---|
| 120 | ` ` | U+0020 | `␣` | `` | ार को  मुला� |
| 3 | `  ` | U+0020 U+0020 | `␣␣` | `` | ्यपि   कैर |

### English

**NFKC**

| count | text | code points | original tokens | re-encoded tokens | context |
|---|---|---|---|---|---|
| 2 | ` of Medical` | U+00A0 U+006F U+0066 U+00A0 U+004D U+0065 U+0064 U+0069 U+00 | `  · of ·   · Medical` | `␣of · ␣Medical` | F enquiry, Department of Medical Education officials are not |
| 1 | ` prayer` | U+00A0 U+0070 U+0072 U+0061 U+0079 U+0065 U+0072 | `  · pr · ayer` | `␣prayer` |  nondenominational prayer sessions. The Camp |

**whitespace**

| count | text | code points | original tokens | re-encoded tokens | context |
|---|---|---|---|---|---|
| 6 | ` ` | U+0020 | `␣` | `` | hanteras on  Krishna Trayodashi |
| 2 | ` of Medical` | U+00A0 U+006F U+0066 U+00A0 U+004D U+0065 U+0064 U+0069 U+00 | `  · of ·   · Medical` | `␣of · ␣Medical` | F enquiry, Department of Medical Education officials are not |
| 1 | ` prayer` | U+00A0 U+0070 U+0072 U+0061 U+0079 U+0065 U+0072 | `  · pr · ayer` | `␣prayer` |  nondenominational prayer sessions. The Camp |

**webform**

| count | text | code points | original tokens | re-encoded tokens | context |
|---|---|---|---|---|---|
| 6 | ` ` | U+0020 | `␣` | `` | hanteras on  Krishna Trayodashi |
| 2 | ` of Medical` | U+00A0 U+006F U+0066 U+00A0 U+004D U+0065 U+0064 U+0069 U+00 | `  · of ·   · Medical` | `␣of · ␣Medical` | F enquiry, Department of Medical Education officials are not |
| 1 | ` prayer` | U+00A0 U+0070 U+0072 U+0061 U+0079 U+0065 U+0072 | `  · pr · ayer` | `␣prayer` |  nondenominational prayer sessions. The Camp |
