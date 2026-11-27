# GPU runbook: watermarked generation, trimming, verification

**One-command job:** after steps 0, 0b and 1, `promptset/job/run_study.sh` (SLURM or nohup) runs steps 2-5
below end to end into one resumable dump, with human-text FPR, per-token detector state and a report; see
[JOB.md](JOB.md). The steps below are what it runs, for doing them by hand.

Scripts in `promptset/scripts/`: `generate.py` (Qwen3-8B with SynthID-Text through HF `generate`),
`trim.py` (one trimming rule for AI and human text), `verify_generations.py` (z-test on the passages).
All three import the repo's `synthid/synthid_wm` library directly; no `pip install -e` is needed
(`_common.add_synthid_to_path()` puts `../../synthid` on `sys.path`; `PYTHONPATH=synthid` works too).
Commands below run from the repo root.

## 0. Setup (once per server)

```bash
git clone <repo-url> indic-watermarking && cd indic-watermarking
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r synthid/requirements.txt          # torch, transformers>=4.51, accelerate, pyyaml, scipy, pytest
# If the default torch wheel does not match the server's CUDA driver, install a matching one first, e.g.
#   pip install torch --index-url https://download.pytorch.org/whl/cu124
python -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python -m pytest promptset/tests -q              # CPU tests, tiny random Qwen3, ~15 s
```

Model weights: Qwen/Qwen3-8B is Apache-2.0 and not gated, so `huggingface-cli login` is optional (it
raises rate limits). Fetch it once so the shard processes do not all download at the same time:

```bash
export HF_HOME=/data/hf                           # optional: a disk with ~20 GB free
huggingface-cli download Qwen/Qwen3-8B            # ~16 GB (newer huggingface_hub: `hf download Qwen/Qwen3-8B`)
# offline servers: copy the snapshot and pass --model /path/to/Qwen3-8B (generate.py) and
# --tokenizer /path/to/Qwen3-8B (trim.py, verify_generations.py)
```

## 0b. Finish the translation and build the G prompt file (before any generation)

`promptset/data/G_prompts*.jsonl` is committed, built from a PARTIAL translation: `promptset/data/work/translated.jsonl`
has 325 of 517 rows (Colab run; all Hindi->English done, 192 English->Hindi essay/personal rows missing), so it holds
301 pairs and essay/personal are short. If the remaining rows were translated elsewhere (Colab notebook, which
resumes from that file), put the finished file there and skip to the build. Otherwise finish it here; then rebuild.

IndicTrans2 needs its own venv (it breaks on transformers 5.x) and a Hugging Face token from an account that
has accepted the terms of `ai4bharat/indictrans2-en-indic-1B` and `ai4bharat/indictrans2-indic-en-1B`:

```bash
python3 -m venv .venv-mt && .venv-mt/bin/pip install torch "transformers==4.57.6" "IndicTransToolkit==1.1.1" sentencepiece sacremoses pyyaml
.venv-mt/bin/huggingface-cli login                 # read token of the account with IndicTrans2 access
W=promptset/data/work
# Resumes: the rows already in translated.jsonl are skipped. Same model (1B) and beams (5) as the existing rows.
.venv-mt/bin/python promptset/scripts/translate_indictrans2.py --in $W/to_translate.jsonl \
    --out $W/translated.jsonl --size 1B --batch 32 --beams 5 --device cuda
wc -l $W/translated.jsonl                          # expect 517
# review the new rows (flag/drop only) into $W/translation_review.jsonl before building, as for the first 325
```

Optional speed-up: `--kv-cache` turns on the beam-search cache. With transformers 4.57.6 IndicTrans2's model
code may crash with it; only use it if a 20-row test (`--limit 20`, two separate `--out` files) gives
identical translations with and without it.

Then build and validate the prompt file (main venv):

```bash
python promptset/scripts/build_promptset.py --work $W --config promptset/config/promptset.yaml \
    --kit promptset/config/ai_levels_prompts.json --out promptset/data
python promptset/scripts/validate_promptset.py promptset/data/G_prompts.jsonl \
    --tokenizer Qwen/Qwen3-8B --report promptset/data/validation.md --strict
```

`build_promptset.py` prints "Missing translations: 0" when everything is in. Translations are used exactly
as IndicTrans2 produced them. `$W/translation_review.jsonl` (optional, `id, ok, drop, issue`) only flags or
drops pairs; it covers the rows reviewed so far, and the validator's `tr.*` warnings (length ratio, Latin
leftovers, repeat loops, identical text) cover the rest. Use `P=promptset/data/G_prompts.jsonl` below.

## 1. Private watermark keys (once for the whole study)

The keys in `promptset/config/generation.yaml` are the PUBLIC demo keys from the HF blog post. Make
private ones and keep them out of git:

```bash
(cd synthid && python -m demo keys --depth 30)    # prints: watermark: {keys: [...]}
cp promptset/config/generation.yaml ~/wm/generation.private.yaml
# paste the printed list over watermark.keys in ~/wm/generation.private.yaml
export RUNCFG=~/wm/generation.private.yaml
```

Never change the `watermark` or `generation` blocks after the pilot: `generate.py` refuses to append to an
output written with a different config (`config_sha`). Each output file gets a sidecar
`<out stem>.config.yaml` with the exact effective config (including the keys: keep it private too);
verification reloads it.

## 2. Plan and pilot

```bash
P=promptset/data/G_prompts.jsonl                  # built in step 0b (edit levels: promptset/data/edit/edit_prompts.jsonl)
mkdir -p runs trimmed scores

# Batch plan only (loads the tokenizer, not the weights): number of batches and padded token slots
python promptset/scripts/generate.py --config $RUNCFG --prompts $P --out runs/pilot.jsonl \
    --filter split=pilot --dry-run

# Pilot, both languages, watermarked (rows with "watermark": false are generated without it)
python promptset/scripts/generate.py --config $RUNCFG --prompts $P --out runs/pilot.jsonl --filter split=pilot
# Matched control run without the watermark (separate file!)
python promptset/scripts/generate.py --config $RUNCFG --prompts $P --out runs/pilot_nowm.jsonl \
    --filter split=pilot --no-watermark

python promptset/scripts/trim.py gen --in runs/pilot.jsonl --out trimmed/pilot.jsonl --tokenizer Qwen/Qwen3-8B
python promptset/scripts/trim.py gen --in runs/pilot_nowm.jsonl --out trimmed/pilot_nowm.jsonl --tokenizer Qwen/Qwen3-8B
python promptset/scripts/verify_generations.py --config runs/pilot.config.yaml --in trimmed/pilot.jsonl \
    --out scores/pilot.jsonl
python promptset/scripts/verify_generations.py --config runs/pilot.config.yaml --in trimmed/pilot_nowm.jsonl \
    --out scores/pilot_nowm.jsonl
```

Check in the pilot: the stderr throughput line (`total N tok/s`), `finish_reason` (share of `length`
per level), `trimmed/pilot.summary.md` (acceptance per language x level x cutoff),
`trimmed/pilot.cleaned.jsonl` (what the cleaning removed), `scores/pilot.summary.md` (TPR) and
`scores/pilot_nowm.summary.md` (FPR). If the token budget OOMs, the log says `CUDA OOM at batch size
...`; lower `batching.token_budget` in `$RUNCFG` (this does not change `config_sha`).

## 3. Full run, per language, several GPUs

One process per GPU, each with its own output file; `--shard i/n` is 0-based and partitions the ids by a
stable hash, so the shards are disjoint and together cover everything, whatever the filters.

```bash
LANG_=hi; N=4
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i nohup python promptset/scripts/generate.py --config $RUNCFG --prompts $P \
      --filter split=main --filter language=$LANG_ --shard $i/$N --out runs/main_$LANG_.s$i.jsonl \
      > runs/main_$LANG_.s$i.log 2>&1 &
done
wait
```

Same for `LANG_=en`, and for the reserve split if needed (`--filter split=reserve`). A crashed or killed
process is resumed by running the same command again: rows already in `--out` (same id, attempt and
watermark flag) are skipped, and a half-written last line is dropped. `--filter` can be repeated (same
key = OR, e.g. `--filter level=G --filter level=C_continue`, or `level=G,C_continue`); `--limit N` takes
the first N selected rows.

## 4. Trim, regenerate failures, repeat

```bash
python promptset/scripts/trim.py gen --in runs/main_hi.*.jsonl --out trimmed/main_hi.jsonl \
    --tokenizer Qwen/Qwen3-8B
# -> trimmed/main_hi.regen_ids.txt: ids whose LARGEST cutoff failed for length (too_short /
#    no_sentence_end / empty), and main_hi.regen_ids.attempt<k>.txt split by the attempt to use next

K=1
python promptset/scripts/generate.py --config $RUNCFG --prompts $P \
    --only-ids trimmed/main_hi.regen_ids.attempt$K.txt --attempt $K --out runs/main_hi.regen$K.jsonl
# (several GPUs: add --shard i/n and per-shard --out files as above)

# re-trim over ALL attempts: per cutoff, the first accepted attempt is used
python promptset/scripts/trim.py gen --in runs/main_hi.*.jsonl --out trimmed/main_hi.jsonl --tokenizer Qwen/Qwen3-8B
# repeat with K=2, 3 ... until the regen list is empty or the attempt cap you set is reached
```

Attempt k > 0 uses a new seed derived from the row's `generation.seed` and k. Script failures
(`script`: Devanagari share < 0.8 for hi, Latin share < 0.9 for en) are not regenerated by default; add
them with `--regen-reasons too_short,no_sentence_end,empty,script`. Trim watermarked and control
(`--no-watermark`) outputs into separate files.

Human passages (rows `id, language, text, ...`), same rule, no cleaning:

```bash
python promptset/scripts/trim.py human --in work/human_hi.jsonl --out trimmed/human_hi.jsonl \
    --tokenizer Qwen/Qwen3-8B --words 100 200 400 800 1200 --tokens 200 400 800 1500
```

Without `--words`/`--tokens`, human mode uses each row's `cutoffs` (the edit builder's
`human_passages.jsonl` has them), else `cutoffs` from `promptset/config/promptset.yaml`.

More trimming rules (all recorded per row and in `*.summary.json` / `.md`):

* **`sentence_end_rule`** per row: `punct` (default) or `punct+linebreak` (hindi-aesthetics, where
  punctuation was stripped upstream): every line break is also a sentence end. Honoured in both modes.
* **C_continue rows** (gen mode): the scored passage is `head` + separator + cleaned continuation (newline for
  `punct+linebreak` rows, else a space). Rows get `ai_char_start`, `ai_words`/`ai_word_share` and
  `ai_tokens`/`ai_token_share`; small cutoffs can fall inside the human head (share 0). `--no-join-head`
  trims the continuation alone.
* **Acceptance rule (default: 95 % or one sentence of slack)**: a passage of n units cut at N is accepted
  iff `n >= min(min_frac * N, N - slack)`. From `config/promptset.yaml`: `min_frac: 0.95`,
  `slack_words: 30`. Word cutoffs get 30 words of slack; token cutoffs get 30 words' worth of tokens in the
  row's language, `round(30 * tokens_per_word[lang])` = **38 (en) / 137 (hi)**. The 95 % rule still governs
  long cutoffs (word N = 800 and 1200: 760 and 1140), the slack the short ones (N = 100 words: 70; hi
  N = 200 tokens: 63). Overrides: `--slack-words S` (also the base of the derived token slack),
  `--slack-tokens T` (T tokens for every language), `--min-frac F`. **`--slack-words 0` turns the slack
  off** (pure 95 % rule). Each row records its `slack` and `min_units`; the summary states the rule. The
  edit builder (`build_edit_prompts.py`) cuts the human inputs with the same rule and the same defaults.
  Why: the 95 % window alone is 5 words at N = 100, and Hindi sentences are long. On all 1,544 eligible
  source documents (samalochan 576, hindi-aesthetics 968; Qwen3 tokens), share accepted, pure 95 % ->
  default:

  | cutoff | samalochan | hindi-aesthetics |
  |---|---|---|
  | 100 words | 28 % -> 88 % | 41 % -> 95 % |
  | 200 words | 48 % -> 87 % | 66 % -> 95 % |
  | 400 words | 79 % -> 91 % | 88 % -> 95 % |
  | 800 / 1200 words | 94 % / 97 % (unchanged) | 95 % / 81 % (unchanged) |
  | 200 tokens | 12 % -> 88 % | 16 % -> 95 % |
  | 400 tokens | 26 % -> 89 % | 30 % -> 95 % |
  | 800 tokens | 39 % -> 83 % | 54 % -> 95 % |
  | 1500 tokens | 65 % -> 86 % | 83 % -> 95 % |

  Most remaining rejections at large N are documents shorter than N.

## 5. Verify

```bash
python promptset/scripts/verify_generations.py --config runs/main_hi.s0.config.yaml \
    --in trimmed/main_hi.jsonl --out scores/main_hi.jsonl                       # TPR
python promptset/scripts/verify_generations.py --config runs/main_hi.s0.config.yaml \
    --in trimmed/human_hi.jsonl --out scores/human_hi.jsonl                     # FPR, same keys
```

Only accepted passages are scored unless `--include-rejected`. Verification needs only the tokenizer and
runs on CPU. Outputs: per-passage `z, p_value, n_scored, decision` and `scores/*.summary.md` by language x
level x cutoff_kind x cutoff x watermarked. `--score-ids` on a generate.py file additionally scores the
sampled token ids, which shows how much is lost to re-tokenisation.

## 6. Token budgets (TODO: fill after the pilot)

Upper bounds known now (sum of `max_new_tokens`, i.e. every row running to its cap; real use is lower because
generation stops at end of text). G rows are computed from the selection and config, so they hold once the
translation is complete:

| file | split | language | rows | mean max_new_tokens | sum max_new_tokens |
|---|---|---|---|---|---|
| G_prompts | pilot | en | 24 | 1,440 | 34,560 |
| G_prompts | pilot | hi | 24 | 5,088 | 122,112 |
| G_prompts | main | en | 376 | 1,444 | 542,912 |
| G_prompts | main | hi | 376 | 5,102 | 1,918,400 |
| G_prompts | reserve | en | 117 | 1,471 | 172,160 |
| G_prompts | reserve | hi | 117 | 5,202 | 608,576 |
| edit_prompts | main | en | 5,120 | 1,188 | 6,085,120 |
| edit_prompts | main | hi | 5,335 | 3,679 | 19,627,776 |

Per-GPU throughput comes from the pilot log (`total N tok/s`); GPU-hours = generated tokens / throughput / 3600.
Devanagari costs several times more Qwen3 tokens than English for the same amount of text (see
`retokenise-experiment/`), so the same `target_words` costs far more tokens in Hindi: take tokens per word
from the pilot (`n_new_tokens` vs `n_words` in the trimmed output).

| split | language | rows | mean max_new_tokens | expected new tokens | tok/s per GPU | GPU-hours |
|---|---|---|---|---|---|---|
| pilot | en | TODO | TODO | TODO | TODO | TODO |
| pilot | hi | TODO | TODO | TODO | TODO | TODO |
| main | en | TODO | TODO | TODO | TODO | TODO |
| main | hi | TODO | TODO | TODO | TODO | TODO |
| regen (attempts 1-3) | both | TODO | TODO | TODO | TODO | TODO |
| control (`--no-watermark`) | both | TODO | TODO | TODO | TODO | TODO |

`batching.token_budget` (default 160,000 for an 80 GB card) bounds `rows x (longest prompt + max_new_tokens)`
per batch; the Qwen3-8B KV cache is ~144 KiB per token in bf16, and SynthID adds a few GB of temporaries.

## Caveats

* **HF `generate` only.** The watermark is a transformers logits processor
  (`watermarking_config=PortableSynthIDConfig`). vLLM, TGI, SGLang or llama.cpp would produce
  unwatermarked text; do not swap the backend for speed.
* **`enable_thinking=False`.** Qwen3's chat template is called with `enable_thinking=False` (config default;
  a row's `chat_template_kwargs` can override it), which inserts an empty `<think></think>` block so the model
  answers directly. Any `<think>...</think>` still present in the output is stripped and flagged `had_think`.
* **Reproducibility needs the same batches.** Each batch is seeded with `torch.manual_seed(hash(row seeds, attempt))`,
  but sampling is batched: a row's text depends on which rows share its batch, so an exact rerun needs the same
  prompts, filters, `--batch-size`/`--token-budget`, no OOM split (a split re-seeds both halves), the same
  GPU model and the same torch/transformers versions (bf16 kernels differ between them). The output records
  `seed_effective`, `batch_seed` and `batch_size`; `<out stem>.runs.jsonl` records versions, GPU and argv.
* **Same sampling everywhere.** temperature 0.7, top_k 40, top_p 1.0 for all levels and both languages;
  Qwen3's own `generation_config` defaults (0.6 / 20 / 0.95) are overridden.
* **Tokenizer and keys must match** between generation, trimming (token cutoffs) and verification.
* **`--attempt` and `--shard` are 0-based**; attempt 0 is the first generation.
* The SynthID processor keeps a 1,024-token context history; the verifier masks every repeated context in the
  whole text, which is slightly conservative for passages over ~1,000 tokens.
