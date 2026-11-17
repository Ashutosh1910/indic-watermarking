"""Fast checks that need no model download.  Run:  .venv/bin/python -m pytest tests -q"""

import math
import random

import pytest
import torch
from transformers import SynthIDTextWatermarkLogitsProcessor

from synthid_wm import DetectionConfig, PortableSynthIDConfig, SynthIDVerifier, WatermarkConfig, build_processor


class FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [int(t) for t in text.split()]}


def _base_processor(wm, device):
    return SynthIDTextWatermarkLogitsProcessor(
        ngram_len=wm.ngram_len, keys=wm.keys, sampling_table_size=wm.sampling_table_size,
        sampling_table_seed=wm.sampling_table_seed, context_history_size=wm.context_history_size,
        device=torch.device(device))


ACCEL = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else None)


@pytest.mark.skipif(ACCEL is None, reason="needs MPS or CUDA")
def test_stock_table_depends_on_device():
    """The reason PortableSynthIDConfig exists: stock transformers draws a different table per device."""
    wm = WatermarkConfig()
    assert not torch.equal(_base_processor(wm, "cpu").sampling_table, _base_processor(wm, ACCEL).sampling_table.cpu())


@pytest.mark.skipif(ACCEL is None, reason="needs MPS or CUDA")
def test_portable_processor_matches_cpu():
    wm = WatermarkConfig()
    cpu = build_processor(wm, "cpu")
    acc = PortableSynthIDConfig.from_watermark_config(wm).construct_processor(151936, torch.device(ACCEL))
    assert acc.sampling_table.device.type == ACCEL
    assert torch.equal(cpu.sampling_table, acc.sampling_table.cpu())
    ids = torch.randint(1, 150_000, (2, 300))
    assert torch.equal(cpu.compute_g_values(ids), acc.compute_g_values(ids.to(ACCEL)).cpu())


@pytest.mark.parametrize("method", ["weighted_mean", "mean"])
def test_null_is_calibrated_on_random_tokens(method):
    """Random token sequences carry no watermark: z ~ N(0, 1), false positives ~ alpha."""
    rng = random.Random(0)
    v = SynthIDVerifier(WatermarkConfig(), FakeTokenizer(), DetectionConfig(method=method, alpha=0.05))
    zs = [v.verify_ids([rng.randrange(1, 150_000) for _ in range(200)]).z for _ in range(400)]
    mean = sum(zs) / len(zs)
    sd = math.sqrt(sum((z - mean) ** 2 for z in zs) / (len(zs) - 1))
    fpr = sum(z > 1.6449 for z in zs) / len(zs)
    assert abs(mean) < 0.2 and 0.85 < sd < 1.15 and fpr < 0.09


def test_high_g_tokens_are_detected_and_wrong_keys_are_not():
    """Greedily pick, from a few candidates, the token with the most g=1 -- a crude watermark."""
    wm = WatermarkConfig()
    proc = build_processor(wm, "cpu")
    rng = random.Random(1)
    ids = [rng.randrange(1, 150_000) for _ in range(wm.ngram_len - 1)]
    for _ in range(150):
        cands = [rng.randrange(1, 150_000) for _ in range(4)]
        g = proc.compute_g_values(torch.tensor([[*ids[-(wm.ngram_len - 1):], c] for c in cands]).reshape(4, -1))
        ids.append(cands[int(g[:, 0, :].sum(-1).argmax())])
    right = SynthIDVerifier(wm, FakeTokenizer()).verify_ids(ids)
    wrong = SynthIDVerifier(WatermarkConfig(keys=[k + 1 for k in wm.keys]), FakeTokenizer()).verify_ids(ids)
    assert right.decision == "watermarked" and right.z > 8
    assert wrong.decision != "watermarked" and abs(wrong.z) < 4


def test_eos_cut_and_insufficient():
    v = SynthIDVerifier(WatermarkConfig(), FakeTokenizer(), DetectionConfig(min_scored=50))
    r = v.verify_ids([5, 6, 7, 8, 9, 10, 0, 11, 12, 13])
    assert r.n_tokens == 6 and r.n_scored == 2 and r.decision == "insufficient"
    assert v.verify_ids([]).decision == "insufficient"


def test_repeated_contexts_are_masked():
    v = SynthIDVerifier(WatermarkConfig(ngram_len=3), FakeTokenizer())
    r = v.verify_ids([1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert r.n_scored == 3 and r.n_masked == 4
