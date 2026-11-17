# SynthID-Text: generate and verify

Watermark text from a Hugging Face model with **SynthID-Text** and check whether a text carries the
watermark. Built on transformers' own `SynthIDTextWatermarkingConfig` and
`SynthIDTextWatermarkLogitsProcessor`, following the
[Hugging Face SynthID-Text post](https://huggingface.co/blog/synthid-text). Default model: Qwen3-8B;
any causal LM id works (use `Qwen/Qwen3-0.6B` for quick runs).

```
synthid/
├── config/default.yaml   model, watermark (keys, ngram_len, ...), generation and detection settings
├── synthid_wm/           library: config.py, watermark.py (load + generate), verify.py (detector)
├── demo/                 CLI: python -m demo {generate, verify, keys}
├── scripts/smoke_test.py end-to-end check with a real model
└── tests/                fast checks, no model download
```

## Setup (macOS or Linux)

```bash
cd synthid
./setup.sh          # creates .venv/ (Python 3.13 via uv if available) and installs requirements.txt
```

No weights are downloaded by setup. A model downloads on first use into the standard Hugging Face
cache (`~/.cache/huggingface`, or `$HF_HOME`). Qwen3-0.6B is about 1.2 GB; Qwen3-8B is about 16 GB and
needs roughly that much free unified memory in bfloat16. Devices are picked automatically
(CUDA, then MPS on Apple Silicon, then CPU); override with `--device` and `--dtype`.

## Demo CLI

```bash
# Generate with the watermark and verify the answer (add --compare for an unwatermarked answer too)
.venv/bin/python -m demo generate --model Qwen/Qwen3-0.6B --prompt "Explain how monsoon rains form."

# Interactive: omit --prompt
.venv/bin/python -m demo generate --model Qwen/Qwen3-0.6B

# Your own watermark: fresh keys, different n-gram length; save the exact settings for later
.venv/bin/python -m demo keys --depth 30 > mykeys.yaml
.venv/bin/python -m demo generate --model Qwen/Qwen3-8B --keys @mykeys.yaml --ngram-len 4 \
    --prompts-file prompts.txt --out outputs/run.jsonl --save-config outputs/run.yaml

# Verify later, in any process, on any device: needs the tokenizer only, not the weights
.venv/bin/python -m demo verify --config outputs/run.yaml --jsonl outputs/run.jsonl
.venv/bin/python -m demo verify --config outputs/run.yaml --file edited.txt --text "some text" --json
```

`generate` prints two verdicts per answer: on the **sampled token ids** and on the **decoded text
re-tokenised** from scratch. A verifier in the wild only ever sees the second; any gap between the two
is retokenisation drift (see `../retokenise-experiment/`).

Flags (all override `--config`): `--model --device --dtype`, watermark `--keys --ngram-len
--table-size --table-seed --context-history --skip-first-ngram-calls`, generation `--max-new-tokens
--temperature --top-k --top-p --seed --greedy --no-chat-template --thinking --no-watermark
--compare`, detection `--method --alpha --min-scored --bayesian-detector`.

## Library

```python
from synthid_wm import load_config, WatermarkedGenerator, SynthIDVerifier, load_tokenizer

cfg = load_config("config/default.yaml", {"model.model_id": "Qwen/Qwen3-0.6B"})
out = WatermarkedGenerator(cfg).generate(["Write about the sea."])[0]

verifier = SynthIDVerifier(cfg.watermark, load_tokenizer(cfg.model.model_id), cfg.detection)
r = verifier.verify_text(out.text)   # VerifyResult: decision, z, p_value, score, n_scored, ...
```

## How verification works

Score-based, no training. For each token whose `ngram_len − 1` preceding tokens are known, transformers'
`compute_g_values` hashes the n-gram with each of the D keys into the sampling table, giving D bits
(g-values). In text the watermark did not touch, these bits are coin flips with P(1) = p0, the fraction
of ones in the sampling table (0.4996 for the defaults). Tournament sampling pushes them above p0.

* Positions whose context already occurred earlier in the text are dropped (transformers'
  `compute_context_repetition_mask`; generation does not watermark them either). Everything after an
  end-of-sequence token is ignored.
* The statistic is the SynthID paper's **weighted mean g-value**: depth weights fall linearly from 10 to
  1. `--method mean` weights all depths equally.
* Under the null, the score has mean p0 and variance p0(1−p0)·Σw²/(D²N) for N scored positions. The
  verifier reports z and the one-sided p-value and calls the text **watermarked when p < alpha**
  (default 0.01). Below `min_scored` positions (default 50) the decision is **insufficient**.
* Optional: `--bayesian-detector PATH` also reports the posterior from a trained transformers
  `BayesianDetectorModel`. Training one is out of scope here; see
  [transformers-research-projects/synthid_text](https://github.com/huggingface/transformers-research-projects/tree/main/synthid_text).

The verifier must use the same `watermark` block **and the same tokenizer** as generation.

### A device bug this code works around

transformers draws the sampling table with `torch.Generator(device=...)` on the device generation runs
on. On a Mac, the MPS table differs from the CPU table for the same seed, so stock
`SynthIDTextWatermarkingConfig` output from MPS does not verify on CPU (z ≈ 0, check (d) below).
`PortableSynthIDConfig` subclasses it and always draws the table on CPU before moving it to the device,
so the watermark depends only on the config. It is still passed as `watermarking_config=`, so
transformers applies it after temperature, top-k and top-p, as usual.

## Checks

```bash
.venv/bin/python -m pytest tests -q          # no downloads: device fix, null calibration, masking
.venv/bin/python scripts/smoke_test.py       # real model, default Qwen/Qwen3-0.6B
```

Smoke test on an M-series Mac (MPS, bfloat16), Qwen3-0.6B, 200 new tokens, default config:

| check | detected | z |
|---|---|---|
| (a) watermarked, scored on sampled ids | 5/5 | 7.9, 5.1, 6.7, 3.3, 5.7 |
| (a) watermarked, re-tokenised text, CPU verifier | 5/5 | same |
| (b) watermarked text, wrong keys | 0/5 | −0.6 to 2.0 |
| (c) unwatermarked generations | 0/5 | −2.4 to 1.3 |
| (c) human Hindi, 20 passages of 200 tokens | 0/20 | −1.6 to 2.0 |
| (d) stock transformers config on MPS, verified on CPU | 0/5 | 0.1 to 1.4 |

Watermark strength depends on entropy: low temperature, small top-k, short answers and repetitive text
all lower z. Hindi generation has not been measured systematically yet: a single 150-token Hindi
answer from Qwen3-0.6B (20 keys, `ngram_len` 4) scored z = 2.2, below the default threshold. The
smoke-test numbers are deterministic for a given seed and device. Qwen3-8B has not been run yet.
