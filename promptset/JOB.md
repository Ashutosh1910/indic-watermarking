# The GPU job: one command from prompt files to an analysable dump

`scripts/run_study.py` runs the whole study on a GPU server and writes everything later analysis needs into
one directory (the **dump**). `job/run_study.sh` wraps it for SLURM or a plain `nohup` run. It drives the
scripts described in [GPU_RUNBOOK.md](GPU_RUNBOOK.md) (`generate.py`, `trim.py`, `build_edit_prompts.py
build-outline`, `verify_generations.py`) and `scripts/load_dump.py` reads the dump back.

## On Sharanga (BITS Hyderabad HPC)

Everything large goes on `/scratch/$USER/wm` (`job/sharanga.env`): the repo, the venv, the Hugging Face cache
(Qwen3-14B is about 30 GB) and the dumps. `$HOME` (40 GiB, backed up, nearly full) holds only the private keys
(`~/wm/keys.private.yaml`) and a copy of each dump's report, results and analysis (`~/wm/results/<dump>/`).

```bash
# login node: downloads only (code must not run on login nodes)
mkdir -p /scratch/$USER/wm && cd /scratch/$USER/wm
git clone https://github.com/Ashutosh1910/indic-watermarking && cd indic-watermarking
bash promptset/job/setup_sharanga.sh             # python 3.11 (conda if needed), venv, packages, Qwen3-14B weights

# the study (one job at a time: the QoS allows 2 GPUs and 12 CPUs per user in total)
WM_TEST_ONLY=1 WM_TIME=03:00:00 bash promptset/job/submit.sh --dump dump/pilot   # estimated start, submits nothing
WM_TIME=03:00:00 bash promptset/job/submit.sh --dump dump/pilot                   # check + pilot, 2x H200
grep -E "PRECHECK|ALL OK|exit" logs/wm-study.<jobid>.out                          # "== PRECHECK OK" first
bash promptset/job/submit.sh --dump dump/main --set filters='[split=pilot,main]'  # full run, after the pilot
squeue -u $USER; tail -f logs/wm-study.<jobid>.out
```

* **Partitions and GPUs.** `submit.sh` sets partition, QoS, GPUs, CPUs and memory together. Default `gpu_h200_8`
  (H200 141 GB, `qos_gpu_h200`, 1-day limit, `generate.token_budget=250000` set for you) with `WM_GPUS=2`:
  Qwen3-14B fits on one card, so each GPU runs its own generation process and time falls about linearly with GPUs.
  `WM_GPUS=1` changes it; the dump resumes with any count. Others: `WM_PARTITION=gpu_h100_4` (H100 80 GB) or
  `WM_PARTITION=gpu_a100_8` (A100 80 GB). One node per job.
* **QoS limits** (`sacctmgr show qos qos_gpu_h200 format=Name,MaxTRESPU%30,MaxJobsPU -p`): per user 2 GPUs and 2
  running jobs; on H100 and H200 also 12 CPUs and 300G in total. `submit.sh` asks for 6 CPUs and 48G per GPU, at most
  12 CPUs (`WM_MAX_CPUS`) and 300G, with the generic `gpu:N` request (the typed H100 name is capped at 1 GPU). A job
  that would go over waits with `QOSMax...PerUser`; to queue the study behind the check job, use
  `WM_DEPENDENCY=afterok:<check jobid>`.
* **Starting sooner.** A shorter `WM_TIME` can start earlier by backfill into a gap before a reserved node.
  `WM_TEST_ONLY=1` shows Slurm's estimate for the exact request; try a few `WM_TIME` values. A job that hits its
  limit resubmits itself and resumes, so a short limit costs only a requeue.
* **Precheck.** The first job of each submit.sh chain starts with `job/precheck.sh` (a few minutes): GPUs and
  driver, internet (for information), and a short watermarked generation with the study model from the offline
  cache (`gpu_check.py`). If it fails, the job stops with `== PRECHECK FAILED` and exit 2, before the study
  starts and without resubmitting. `WM_PRECHECK=tests` also runs the CPU test suites (about 15 min);
  `WM_PRECHECK=0` skips the check. `sbatch promptset/job/check_sharanga.sh` runs the check with the tests as a
  job of its own (1 GPU).
* **Monitoring.** Everything is in files, so the login node needs only `tail` and `cat`:

  ```bash
  tail -f logs/wm-study.<jobid>.out            # stage lines, and a status block every 15 min while generating
  cat dump/main/status.txt                     # the latest status block
  tail -f dump/main/logs/gen.wm.a0.s*.log      # every batch: rows done/total, tok/s, EOS count, ETA
  ls dump/main/failures/ dump/main/gen/*/*.failures.jsonl   # failed stages and batches
  srun --jobid=<jobid> --overlap nvidia-smi    # GPU memory and use on the job's node
  ```

  The status block (`scripts/watch_study.py`) gives per stream: rows written, per-shard progress, total tok/s
  and ETA; per language: EOS share, mean new tokens, Hindi outputs that are mostly not Devanagari; and a z check
  of the newest 40 watermarked 100%-AI rows and unwatermarked rows. It ends with `STATUS OK` or `STATUS WARN (n)`
  and one `WARN` line per problem: a shard log silent for 30 min, failed batches or stages, under 50 % of outputs
  ending at EOS, over 10 % of Hindi outputs not in Devanagari, `<think>` blocks, watermarked 100%-AI text with
  mean z below 2, or unwatermarked text with mean z above 1.5. The first block comes after the outline stream
  (OX, unwatermarked) and 15 min into the watermarked stream. Settings: `watch` in `config/study.yaml`.
* **Stopping.** `touch /scratch/$USER/wm/STOP` (no resubmission), then `scancel <jobid>`. What was generated
  stays in the dump; the same submit command resumes it after a fix.
* **Walltime and resubmission.** Jobs ask for 23 h 50 min, which every partition allows. Fifteen minutes before
  the limit the job stops cleanly and queues itself again (`--dependency=afterany`), up to `WM_MAX_CHAIN=8` times,
  and the new job resumes the dump. To stop a chain: `touch /scratch/$USER/wm/STOP`, then `scancel` the jobs.
* **Offline.** Inside jobs `HF_HUB_OFFLINE=1`: the weights and tokenizer come from the cache setup filled, so
  compute nodes need no internet (`WM_ONLINE=1` turns this off).
* **Scratch purge.** `/scratch` deletes files 15 days after their last modification. Every job start (and setup)
  touches files under `/scratch/$USER/wm` older than 7 days. If no job runs for over a week, run
  `source promptset/job/sharanga.env && wm_refresh` once, or rerun setup (it skips what is there).
* **Keys.** The first real job creates `~/wm/keys.private.yaml` in your home (private, backed up). Never delete it
  while dumps of the study exist.
* **Driver.** The default torch wheel needs a recent NVIDIA driver; the precheck at the start of the job shows it. If torch reports
  no CUDA, rerun setup with `TORCH_INDEX=https://download.pytorch.org/whl/cu124`.

## Quick start (a fresh GPU server)

```bash
git clone https://github.com/Ashutosh1910/indic-watermarking && cd indic-watermarking
bash promptset/job/setup_gpu.sh          # venv, packages, Qwen3-14B weights, GPU check, CPU tests (once)
source .venv/bin/activate

# 1. what will run, with the token budget per stream and language (starts nothing, needs no GPU)
python promptset/scripts/run_study.py --plan --dump dump/pilot
# 2. pilot: every level in both languages on 8 docs per language + 18 G prompts per language
sbatch promptset/job/run_study.sh --dump dump/pilot              # or: nohup bash promptset/job/run_study.sh --dump dump/pilot > pilot.out 2>&1 &
# 3. read dump/pilot/report.md and dump/pilot/analysis/analysis.md (throughput: generation.csv gpu_seconds)
# 4. full run into its own dump (pilot rows are included again, so main is complete on its own)
sbatch promptset/job/run_study.sh --dump dump/main --set filters='[split=pilot,main]'
```

Nothing else is needed by hand. The job itself:

* **Keys.** `generation.yaml` has the public demo keys. The job swaps in private keys from `~/wm/keys.private.yaml`
  (study.yaml `keys.file`) and creates that file on the first real run (30 random keys, mode 0600). Pilot and
  main then share one key set. Keep the file and never commit it. `RUNCFG=<file>` still works for a run config
  with its own keys.
* **Model.** It downloads the weights once (stage `model`) before the per-GPU processes start.
* **Inputs.** It builds what is missing (stages `inputs:edit`, `inputs:human_pool`) from the corpora under `data/`:
  `work/edit/edit_prompts.jsonl` + `human_passages.jsonl` (300 docs per language, story/essay 50/50, the first 8 per
  language as `split=pilot`) and `work/human_pool.jsonl` (every document, for the FPR check). They hold corpus text
  and stay in `work/` (gitignored). A file the job built is rebuilt only when its settings change. The G prompts
  are in git (`promptset/data/G_prompts.jsonl`, 473 pairs).
* **Analysis.** The last stage runs `analyze.py` on the dump (CPU): see [ANALYSIS.md](ANALYSIS.md).

Budget knobs (estimate before the pilot, about 22M new tokens for the full run, roughly 10-40 A100-hours on
Qwen3-14B; the pilot measures the real throughput): `--set build_inputs.edit.n_docs=200` (fewer docs),
`--set filters='[split=pilot,main, language=en]'` (one language at a time), or
`--set model=Qwen/Qwen3-8B` (faster; a separate dump).

Settings live in `config/study.yaml` (streams, filters, regeneration cap, human sets, z threshold, GPUs);
`--set key=value` overrides any of them. Qwen3-14B wants an 80 GB card; on 48 GB add `--set generate.token_budget=30000`.

**Edit subsample.** `edit_subsample.docs_per_language` (default 200 of the 300 built documents) keeps L1-L4 and
C_continue for that many documents per language, story/essay 50/50, pilot documents first, then a seeded hash
order, the same documents at every level and view. G, G_title, OX/O and the human anchors keep all documents.
`--set edit_subsample.docs_per_language=null` runs all 300. Each 20 documents per language is about 1.5M max new
tokens (5 % of the full run), almost all of it Hindi.

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
| preflight | | private keys, model download, `inputs:edit` / `inputs:human_pool` when missing; `config/run_config.yaml` (keys, private), `config/study.yaml`, input sha256s, environment |
| generate attempt 0 | ox, then wm, ctrl | `gen/<stream>/a0.s<i>of<n>.jsonl`, one process per GPU |
| outline | after ox | `prompts/O_prompts.jsonl` (O_outline rows, fed to wm and ctrl) |
| trim, regenerate | wm, ctrl | `trim/<stream>.jsonl`; ids whose largest cutoff fails for length get attempt k = 1, 2, 3 (new seeds); per cutoff the first accepted attempt is used |
| human | each human set | `trim/human.<set>.jsonl`, every cutoff of promptset.yaml, same rule |
| score | every trimmed file + raw generations | `scores/*.jsonl`, `tokens/*.jsonl.gz` |
| report | | `report.md`, `results/*.csv` |
| analysis | | `analysis/` (analyze.py: tables, figures, `analysis.md`) |

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
