"""Decide whether a text carries a given SynthID-Text watermark.

Method (no training): a one-sided z-test on the (weighted) mean g-value.

For every token at position t >= ngram_len - 1 the verifier hashes the n-gram ending at t with
each of the D watermark keys and looks the hash up in the sampling table, giving D g-values in
{0, 1}. This is transformers' own ``SynthIDTextWatermarkLogitsProcessor.compute_g_values``.

* Without the watermark, the g-values are coin flips that do not depend on the text, so each
  equals 1 with probability p0, the fraction of ones in the sampling table (about 0.5).
* With the watermark, tournament sampling picked tokens with high g-values, so the mean rises
  above p0.

Positions whose (ngram_len - 1)-token context already appeared earlier in the text are masked,
as at generation time (the processor does not watermark a repeated context), using transformers'
``compute_context_repetition_mask``. Text after the first end-of-sequence token is ignored.

The score is the weighted mean from the SynthID paper (Dathathri et al., Nature 2024): depth d gets
weight w_d, linearly from 10 down to 1, rescaled so the weights sum to D. With N scored positions,

    S = sum_{t,d} w_d g_{t,d} / (D N),   E0[S] = p0,   Var0[S] = p0 (1 - p0) sum_d w_d^2 / (D^2 N)

and z = (S - p0) / sqrt(Var0[S]), p = 1 - Phi(z). ``method="mean"`` uses w_d = 1. The text is
called watermarked when p < alpha. With fewer than ``min_scored`` scorable positions the
decision is "insufficient" (z and p are still reported).

The null assumes g-values are independent across positions and depths. That holds well for
distinct n-grams and distinct keys; ``tests/test_verify.py`` checks the calibration on random
token sequences, and the README reports the false-positive rate on human-written text.

Optional: a trained Bayesian detector (transformers' ``BayesianDetectorModel``), if
``detection.bayesian_detector`` points to one. Training it is out of scope here; see
https://github.com/huggingface/transformers-research-projects/tree/main/synthid_text.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch

from .config import DetectionConfig, WatermarkConfig
from .watermark import build_processor


@dataclass
class VerifyResult:
    decision: str            # "watermarked" | "not_watermarked" | "insufficient"
    z: float
    p_value: float
    score: float             # the (weighted) mean g-value the test uses
    mean_g: float            # plain mean g-value over scored positions and all depths
    null_mean: float         # p0
    n_tokens: int            # tokens considered (after cutting at EOS)
    n_scored: int            # positions with a full, non-repeated context
    n_masked: int            # positions skipped because their context repeated
    method: str
    alpha: float
    bayesian_posterior: float | None = None

    @property
    def is_watermarked(self) -> bool:
        return self.decision == "watermarked"

    def to_dict(self) -> dict:
        return asdict(self)


def depth_weights(depth: int, method: str) -> torch.Tensor:
    if method == "mean":
        return torch.ones(depth, dtype=torch.float64)
    w = torch.linspace(10, 1, depth, dtype=torch.float64) if depth > 1 else torch.ones(1, dtype=torch.float64)
    return w * depth / w.sum()


def _norm_sf(z: float) -> float:
    """One-sided upper tail of the standard normal, accurate far into the tail."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


class SynthIDVerifier:
    """Score texts (or token ids) against one watermark config and tokenizer.

    Only the tokenizer is needed, not the model weights::

        tok = load_tokenizer("Qwen/Qwen3-8B")
        v = SynthIDVerifier(cfg.watermark, tok, cfg.detection)
        v.verify_text(text).decision
    """

    def __init__(self, wm: WatermarkConfig, tokenizer, detection: DetectionConfig | None = None,
                 stop_ids: Iterable[int] | None = None):
        self.wm = wm
        self.tokenizer = tokenizer
        self.det = detection or DetectionConfig()
        self.proc = build_processor(wm, "cpu")
        if self.det.null_mean == "table":
            self.p0 = self.proc.sampling_table.double().mean().item()
        else:
            self.p0 = float(self.det.null_mean)
        self.weights = depth_weights(wm.depth, self.det.method)
        if stop_ids is None:
            stop_ids = {i for i in (tokenizer.eos_token_id, tokenizer.pad_token_id) if i is not None}
        self.stop_ids = set(stop_ids)
        self._bayes = None
        if self.det.bayesian_detector:
            self._bayes = self._load_bayesian(self.det.bayesian_detector)

    # ------------------------------------------------------------------ public API

    def verify_text(self, text: str) -> VerifyResult:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self.verify_ids(ids)

    def verify_texts(self, texts: Iterable[str]) -> list[VerifyResult]:
        return [self.verify_text(t) for t in texts]

    def verify_ids(self, ids: list[int] | torch.Tensor) -> VerifyResult:
        ids = list(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
        for i, t in enumerate(ids):
            if t in self.stop_ids:
                ids = ids[:i]
                break
        n_tokens = len(ids)
        n_pos = max(0, n_tokens - (self.wm.ngram_len - 1))
        if n_pos == 0:
            return self._result(0.0, 0.0, 0, 0, n_tokens, None)

        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            g = self.proc.compute_g_values(x)[0].double()                       # [n_pos, D]
            mask = self.proc.compute_context_repetition_mask(x)[0].bool()        # [n_pos]
        n_scored = int(mask.sum())
        if n_scored == 0:
            return self._result(0.0, 0.0, 0, n_pos, n_tokens, None)
        gs = g[mask]
        score = float((gs * self.weights).sum() / (self.wm.depth * n_scored))
        mean_g = float(gs.mean())
        posterior = self._bayes_posterior(x) if self._bayes is not None else None
        return self._result(score, mean_g, n_scored, n_pos - n_scored, n_tokens, posterior)

    def token_details(self, ids: list[int] | torch.Tensor) -> dict:
        """Per-position detector state of a token sequence, for later analysis.

        Returns ``token_ids`` (cut at the first stop id), and for every position t >= ngram_len - 1 (the
        positions ``verify_ids`` considers, in order): ``g_bits`` (the D g-values packed into one int,
        bit d = g-value of key d) and ``mask`` (1 = scored, 0 = masked because its context repeated).
        From these the (weighted) score and z of any prefix can be recomputed exactly; see
        ``promptset/scripts/load_dump.py``'s ``prefix_z``.
        """
        ids = list(ids.tolist() if isinstance(ids, torch.Tensor) else ids)
        for i, t in enumerate(ids):
            if t in self.stop_ids:
                ids = ids[:i]
                break
        out = {"token_ids": ids, "g_bits": [], "mask": [], "depth": self.wm.depth,
               "ngram_len": self.wm.ngram_len, "null_mean": self.p0, "method": self.det.method}
        if len(ids) < self.wm.ngram_len:
            return out
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            g = self.proc.compute_g_values(x)[0].long()                          # [n_pos, D]
            mask = self.proc.compute_context_repetition_mask(x)[0].long()        # [n_pos]
        powers = (2 ** torch.arange(self.wm.depth, dtype=torch.long))
        out["g_bits"] = (g * powers).sum(-1).tolist()
        out["mask"] = mask.tolist()
        return out

    # ------------------------------------------------------------------ internals

    def _result(self, score, mean_g, n_scored, n_masked, n_tokens, posterior) -> VerifyResult:
        D = self.wm.depth
        if n_scored > 0:
            var = self.p0 * (1 - self.p0) * float((self.weights ** 2).sum()) / (D * D * n_scored)
            z = (score - self.p0) / math.sqrt(var)
            p = _norm_sf(z)
        else:
            z, p = 0.0, 1.0
        if n_scored < self.det.min_scored:
            decision = "insufficient"
        else:
            decision = "watermarked" if p < self.det.alpha else "not_watermarked"
        return VerifyResult(decision=decision, z=z, p_value=p, score=score, mean_g=mean_g,
                            null_mean=self.p0, n_tokens=n_tokens, n_scored=n_scored, n_masked=n_masked,
                            method=self.det.method, alpha=self.det.alpha, bayesian_posterior=posterior)

    def _load_bayesian(self, path: str):
        from transformers import BayesianDetectorModel, SynthIDTextWatermarkDetector

        model = BayesianDetectorModel.from_pretrained(path)
        return SynthIDTextWatermarkDetector(model, self.proc, self.tokenizer)

    def _bayes_posterior(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            out = self._bayes(x)
        probs = out[0] if isinstance(out, tuple) else out.posterior_probabilities
        return float(probs.reshape(-1)[0])
