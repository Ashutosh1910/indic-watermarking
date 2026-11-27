# The GPU job: one command from prompt files to an analysable dump

`scripts/run_study.py` runs the whole study on a GPU server and writes everything later analysis needs into
one directory (the **dump**). `job/run_study.sh` wraps it for SLURM or a plain `nohup` run. It drives the
scripts described in [GPU_RUNBOOK.md](GPU_RUNBOOK.md) (`generate.py`, `trim.py`, `build_edit_prompts.py
build-outline`, `verify_generations.py`) and `scripts/load_dump.py` reads the dump back.

## Before the first run (once)

1. Setup, model weights and private keys: GPU_RUNBOOK.md steps 0 and 1 (`export RUNCFG=~/wm/generation.private.yaml`).
2. Prompt files: `promptset/data/G_prompts.jsonl` (runbook step 0b) and `promptset/data/edit/edit_prompts.jsonl`
   plus `human_passages.jsonl` (EDIT_LEVELS.md step 1). A missing prompt file is skipped with a warning.
3. Human pool for the false-positive check (all documents of every human corpus, not only the edit anchors):

   ```bash
   python promptset/scripts/build_human_pool.py --out work/human_pool.jsonl \
       --dir hindisamay:hi:data/HindiSamay --dir samalochan:hi:data/Samalochan --dir medium:en:data/Medium
   ```

   (`data/HindiSamay` and `data/Medium` come from the `add-hindi-corpora` branch. Add
   `--dir hindi-aesthetics:hi:data/Hindi-Aesthetics-Corpus:linebreak` for the punctuation-stripped corpus.)
   The file holds corpus texts, so it stays out of git (`work/` is ignored).

## Run

```bash
export RUNCFG=~/wm/generation.private.yaml
# pilot (the default filters: [split=pilot])
sbatch promptset/job/run_study.sh                              # or: nohup bash promptset/job/run_study.sh > pilot.out 2>&1 &
# full run into its own dump
sbatch promptset/job/run_study.sh --set filters='[split=pilot,main]' --dump dump/main
python promptset/scripts/run_study.py --plan --dump dump/main --set filters='[split=pilot,main]'   # what is left to do
```

Settings live in `config/study.yaml` (streams, filters, regeneration cap, human sets, z threshold, GPUs);
`--set key=value` overrides any of them. On a 40/48 GB card add `--set generate.token_budget=60000`.

**Resume:** submit the same command again. Rows already generated are skipped (per id, attempt and watermark
flag, across all shard files, so the number of GPUs may change); trimming and scoring are redone only when
their inputs changed. A finished dump re-run does nothing. Two jobs on one dump are refused (lock file).
SIGTERM (scancel, preemption, `--signal` before the time limit) stops the generation processes, keeps
what they wrote, and exits 130.

**Integrity:** the first run freezes the watermark keys and sampling settings in `dump/config/run_config.yaml`;
a later run with other keys, sampling or model is refused (use another dump). Detection settings may change.

## What it does

| step | per stream | output |
|---|---|---|
| preflight | | `config/run_config.yaml` (keys, private), `config/study.yaml`, input sha256s, environment |
| generate attempt 0 | ox, then wm, ctrl | `gen/<stream>/a0.s<i>of<n>.jsonl`, one process per GPU |
| outline | after ox | `prompts/O_prompts.jsonl` (O_outline rows, fed to wm and ctrl) |
| trim, regenerate | wm, ctrl | `trim/<stream>.jsonl`; ids whose largest cutoff fails for length get attempt k = 1, 2, 3 (new seeds); per cutoff the first accepted attempt is used |
| human | each human set | `trim/human.<set>.jsonl`, every cutoff of promptset.yaml, same rule |
| score | every trimmed file + raw generations | `scores/*.jsonl`, `tokens/*.jsonl.gz` |
| report | | `report.md`, `results/*.csv` |

Streams (study.yaml): **wm** = watermarked G + edit levels + O; **ctrl** = the same G prompts without the
watermark (AI-text FPR; widen its level list for matched edit controls); **ox** = unwatermarked outline
extraction (not scored). Detection: a text is flagged iff **z > 3** with at least `detection.min_scored`
scored tokens (`flag`; `decision` uses the same threshold, alpha = P(Z > 3)).

The trimming rule is trim.py's: cut at the last sentence end (`. ! ? । ॥ …`) at or before N words or
tokens, accept iff n ≥ min(0.95 N, N − slack) (slack 30 words / en 38, hi 137 tokens). One generation
serves both views: `cutoff_kind` words = same-words, tokens = same-tokens (`experiment_view` in the loader).

## Dump layout

```
dump/<name>/
  manifest.json            last run: status, config_sha, model, tokenizer, GPUs, input sha256s,
                           environment (versions, GPUs, git commit), every stage with status and seconds
  job_runs.jsonl           one line per job run
  config/                  run_config.yaml (WATERMARK KEYS: keep private), study.yaml
  prompts/                 O_prompts.jsonl, <stream>.selected_ids.txt
  gen/<stream>/            a<k>.s<i>of<n>.jsonl   raw generations: the full prompt row (messages, metadata,
                           level, k, genre, source, split, ...) + text, token_ids, n_new_tokens,
                           finish_reason, had_think, attempt, watermarked, seed_effective, batch_seed,
                           batch_size, prompt_tokens, max_new_tokens_effective, config_sha, seconds, gen_time_utc
                           a<k>.s<i>of<n>.runs.jsonl / .config.yaml   per-process versions, GPU, argv, config
  trim/                    <stream>.jsonl        one row per (id, cutoff kind, cutoff): accepted, reject_reason,
                                                 passage, n_words, n_tokens, min_units, slack, attempt,
                                                 attempts_tried, script_share, cleaning, ai_* shares (C rows)
                           <stream>.cleaned.jsonl  text_raw vs text_clean per generation, what cleaning removed
                           <stream>.summary.md/.json, .regen_ids*.txt
                           human.<set>.jsonl      the same for human text
  scores/                  <stream>.jsonl / human.<set>.jsonl   per accepted passage: z, p_value, flag, score,
                                                 mean_g, n_scored, n_masked, n_tokens_verified, + passage metadata
                           raw.<stream>.jsonl     the full raw generation: text z and sampled-ids z (ids_z)
                           *.summary.md/.json
  tokens/                  <stream>.jsonl.gz      per text (longest passage per id/attempt): token_ids,
                                                 g_bits (D g-values packed per position), mask (1 = scored)
  failures/                job.jsonl (stage failures), gen.<stream>.jsonl (failed batches with ids and
                           traceback), dropped.<stream>.jsonl (largest cutoff still rejected after the last
                           attempt), missing.<stream>.txt (selected rows never generated)
  logs/                    one log per stage and GPU process
  report.md, results/      summary tables (generation, acceptance, detection rates, human FPR)
```

The dump contains the watermark keys (`config/`, `gen/*/*.config.yaml`) and corpus text (`trim/human.*`,
edit prompts in `gen/`): do not publish it as is.

## Analysis

```python
import sys; sys.path.insert(0, "promptset/scripts")
from load_dump import Dump, rate_table, prefix_z
d = Dump("dump/main")
res = d.results_df()          # scored passages joined with their trim rows (pandas)
hum = res[res.watermarked == "human"]
fpr = rate_table(hum.to_dict("records"), ["stream", "language", "cutoff_kind", "cutoff"])   # with Wilson CIs
gen = d.generations_df(text=True)                 # raw generations; ids=True adds token_ids
fails = d.failures()                               # dict of lists: job, generate, dropped, missing
for t in d.tokens("wm"):                           # per-token detector state
    z400, n = prefix_z(t, n_tokens=400)            # z of any prefix, exactly the verifier's statistic
```

```bash
python promptset/scripts/load_dump.py report dump/main            # rebuild report.md and results/*.csv
python promptset/scripts/load_dump.py rates dump/main --by language,cutoff_kind,cutoff --where watermarked=human
python promptset/scripts/load_dump.py rates dump/main --z-threshold 4   # other thresholds from the same dump
```

Human ids, edit anchors and pool documents are scored as separate sets (`human.edit_anchors`, `human.pool`),
so a document in both is never counted twice within one set. To compare edit levels at equal amounts of
change, join `results` with the edit prompts on `pair_id` / `passage_id` (the human input is in
`human_passages.jsonl` and in each generation row's prompt).

## Tests

```bash
python -m pytest promptset/tests/test_run_study.py -q   # whole job on CPU with a tiny random Qwen3
```
