# GPU runbook: watermarked generation, trimming, verification

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
P=work/prompts.jsonl                              # prompt JSONL from the prompt-set step
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
* **Slack floor**: `--slack-words S` / `--slack-tokens S` accept a passage when
  `n >= min(min_frac * N, N - S)`. The default 0 keeps the agreed 95 % rule. The 95 % window is narrow at
  small N (5 words at N = 100), and Hindi sentences are long, so many texts have no sentence end inside it.
  On 60 samalochan documents, words only: 28 % accepted at N = 100, 45 % at 200, 85 % at 400, 100 % at 800.

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
