# Prompt set: Hindi and English, model-ready

This folder holds everything up to the GPU step: the prompts for watermarked generation with Qwen3-8B
(SynthID-Text) in English and Hindi, at two kinds of fixed length (same words, same tokens). It also has the
scripts that generate, trim and verify the outputs.

* **G level** (generate from scratch): 517 prompt pairs from existing human-written prompt datasets. Every
  prompt is used in both languages: its native language plus an IndicTrans2 translation.
* **Edit levels** (L1 proofread, L2 polish, L3 paraphrase, L4 rewrite, C continue, OX outline extraction,
  G title): built from the human Hindi corpora (samalochan, hindi-aesthetics). See [EDIT_LEVELS.md](EDIT_LEVELS.md).
* **GPU steps** (translation, generation, trimming, verification): see [GPU_RUNBOOK.md](GPU_RUNBOOK.md).

## Status (2026-11-23)

| step | status |
|---|---|
| Download and rule-filter sources | done |
| LLM keep/genre check of every candidate ([LABELING.md](LABELING.md)) | done, `data/work/labels/` |
| Selection: 517 balanced pairs | done, `data/work/selected.jsonl` |
| IndicTrans2 1B translation | **partial**: 64 of 517 rows (all English→Hindi) were translated on a laptop before it was stopped. They are on that laptop, not in the repo yet. The rest runs on the GPU server (runbook step 0b, resumable). |
| Translation review (flag or drop only, never rewrite) | not done; runs on the finished translations |
| `data/G_prompts*.jsonl` | **not built yet**; built in runbook step 0b once the translation is in |
| Edit-level prompts | done: 4,597 rows from 300 docs, strict validator 0 errors (`data/edit/`; the 85 MB JSONL is gitignored, rebuild with the command in EDIT_LEVELS.md) |
| generate / trim / verify scripts and tests | done, 168 tests pass |

With the full translation, the G file has 1,034 rows (517 prompts × 2 languages). If you build it from a
partial translation, you get only the pairs whose translation exists. English→Hindi runs first, so a partial
file has no Hindi-native prompts, and its genre, tier and pilot balance is off. Rebuilding after the
translation finishes gives the full set; nothing else changes.

## Pipeline

```
download_sources.py  -> work/raw/<source>.jsonl
filter_prompts.py    -> work/candidates/<source>.jsonl, filter_drops.json      (keyword rules)
LLM check            -> work/labels/*.jsonl  (cid, keep, genre, reason)        (LABELING.md)
select_prompts.py    -> selected.jsonl, to_translate.jsonl, selection_report.json
translate_indictrans2.py -> translated.jsonl                                   (GPU step 0b)
build_promptset.py   -> data/G_prompts.jsonl (+ .en / .hi, summary)            (GPU step 0b)
validate_promptset.py   checks every row (lengths, chat template, think block, translation warnings)
build_edit_prompts.py   -> data/edit/edit_prompts.jsonl, human_passages.jsonl
generate.py -> trim.py -> verify_generations.py                                (GPU_RUNBOOK.md)
```

Commands (from `promptset/scripts/`, work dir `../work` is gitignored; the committed copies of the
selection outputs are in `data/work/`):

```bash
python download_sources.py --out ../work          # needs Hugging Face access; PERSUADE comes from Google Drive via gdown
python filter_prompts.py --work ../work
# LLM check: label work/candidates per LABELING.md into work/labels/
python select_prompts.py --work ../work --config ../config/promptset.yaml
python translate_indictrans2.py --in ../work/to_translate.jsonl --out ../work/translated.jsonl --size 1B
python build_promptset.py --work ../work --config ../config/promptset.yaml --kit ../config/ai_levels_prompts.json --out ../data
```

Everything is seeded (`seed: 20261123` in `config/promptset.yaml`). Row ids and generation seeds come from a
hash of the pair id, so they stay the same across rebuilds.

## Sources and licenses

| source | language | what is used | license |
|---|---|---|---|
| WritingPrompts | en | story premises (rendered with the kit's premise template) | MIT |
| PERSUADE 2.0 | en | essay prompts (8 independent ones) | CC BY 4.0 |
| WildChat-1M | en | real user writing requests | ODC-BY |
| No Robots (Generation, Brainstorm) | en | human-written requests | **CC BY-NC 4.0, non-commercial** |
| Dolly 15k | en | creative-writing requests | CC BY-SA 3.0 |
| oasst2 | en | first user turns | Apache 2.0 |
| Aya dataset (hin, original annotations only) | hi | human-written Hindi requests | Apache 2.0 |
| Anudesh (ai4bharat/indic-instruct-data-v0.1) | hi | human-written Hindi requests (English rows go to the English pool) | CC BY 4.0 |
| IndicTrans2 1B (ai4bharat) | en↔hi | machine translation of every prompt into the other language | MIT (gated on HF: accept the terms) |

LMSYS-Chat-1M is excluded (no-redistribution license). Each row carries its `license`. Because No Robots is
NC, a public release of generated text from those rows is non-commercial only. You can drop them with
`--filter`, or by removing the source from `english_source_weights`.

Selected pairs by source: wildchat 241, no_robots 85, anudesh 63, aya 47, writingprompts 31, dolly 24,
oasst2 13, persuade 8, aya_eval_dolly 5.

## What is ours and what is sourced

* **Prompt text is sourced**: every task text is a human-written prompt from the datasets above. No prompt
  was written or translated by Claude.
* **Translations are machine-only**: IndicTrans2 1B, used exactly as produced. Reviews may flag or drop a
  pair (`translation_review.jsonl`: `id, ok, drop, issue`) but never rewrite it.
* **Ours (fixed templates, the same for every row)**: the system prompts and level phrasings in
  `config/ai_levels_prompts.json`, the length clause ("Write about {n} words." / "लगभग {n} शब्दों में
  लिखें।"), and the WritingPrompts premise template. `template_id` in each row says which one was used.
* **Claude's judgement**: the keep/genre labels (LABELING.md); any person or model can redo them from
  that file.

## Selection

* Filters drop short-form prompts (poems, lists, emails, titles, scripts…), prompts that need an input text,
  factual or how-to questions, NSFW and fan-fiction, and prompts with their own length cap. A word target is
  added instead.
* Genres: story, opinion, essay, personal. Each genre has 100 main prompts, 6 of them flagged pilot (3 per tier,
  alternating native language), plus a reserve.
* About half of each genre is Hindi-native where the Hindi pool allows. The Hindi pool is thin: Hindi-native
  pairs are story 75, essay 21, opinion 3, personal 2. So opinion and personal in Hindi are mostly
  translations of English prompts. `native_lang` and `translated` let you analyse native and translated
  prompts separately.
* Length tiers: long (1,300 words requested) and short (400 words). Within each stratum the tiers alternate.

## Row format (`data/G_prompts.jsonl`)

`id, pair_id, language, level="G", genre, source, source_id, native_lang, translated, translation_model,
translation_reviewed, license, split (pilot|main|reserve), length_tier, target_words, max_new_tokens,
cutoffs{words,tokens}, prompt_text, original_text, template_id, messages [system, user],
chat_template_kwargs {enable_thinking: false}, generation {seed}, watermark: true, source_meta`

`messages` goes straight to `tokenizer.apply_chat_template(..., enable_thinking=False)`. Qwen3 then inserts
the empty `<think>\n\n</think>\n\n` block, and the validator checks this.

## Lengths and the trimming rule

* Qwen3 tokens per word: en 1.26, hi 4.57 (`tokens_per_word`).
* Cutoffs: words 100, 200, 400, 800, 1200; tokens 200, 400, 800, 1500. A row serves every cutoff up to
  0.95 × its requested length. `max_new_tokens` = 1.3 × requested tokens, rounded up to 64.
* Generate past the target, then cut at the last sentence end (`. ! ? । ॥`, plus closing quotes) at or before
  N. One generation serves both the same-words and the same-tokens experiments.
* **Acceptance rule** (chosen 2026-11-23, "95% or one sentence of slack"): accept a cut passage of n units
  iff n ≥ min(0.95 × N, N − slack). For word cutoffs the slack is 30 words. For token cutoffs it is 30 words'
  worth of tokens in that language: en 38, hi 137. The 95% rule still governs long cutoffs. `slack_words: 0`
  in `config/promptset.yaml`, or `trim.py --slack-words 0`, restores the pure 95% rule.
* Human passages are trimmed with the same rule (`trim.py human`). On the Hindi corpora, acceptance at
  100 words rose from 28% to 88% (samalochan) and from 41% to 95% (hindi-aesthetics). At small token cutoffs
  a Hindi passage can be well under N (from 63 tokens at N = 200), so analyses should use the actual
  `n_words` / `n_tokens` of each passage, not just N.

## Tests

```bash
python -m pytest promptset/tests -q     # 168 tests, CPU only, tiny random Qwen3; about 6 minutes
```

## Translation notes

* IndicTrans2 needs `transformers==4.57.6` and `IndicTransToolkit==1.1.1` (it breaks on transformers 5.x),
  in its own venv.
* With that version its model code crashes on the beam-search KV cache, so the script defaults to
  `use_cache=False`. That is slow, about 24 s per prompt on a 16 GB Mac, but correct. Use `--kv-cache` only
  after checking that it gives identical translations.
* `notebooks/translate_colab.ipynb` runs the same 1B model on a free Colab or Kaggle T4, and resumes from a
  partial `translated.jsonl`.
* `build_promptset.py` uses a translation only if it was made from exactly the current source text in the
  right direction, so an edited prompt never picks up a stale translation.
