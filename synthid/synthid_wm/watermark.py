"""Load a causal LM and generate SynthID-watermarked text with transformers.

Everything watermark-related is transformers' own code (``SynthIDTextWatermarkingConfig`` and
``SynthIDTextWatermarkLogitsProcessor``), as in https://huggingface.co/blog/synthid-text.
The one change is :class:`PortableSynthIDConfig`, which fixes a device dependence:

transformers draws the random g-value sampling table with ``torch.Generator(device=...)`` on
whatever device generation runs on. CPU, CUDA and MPS generators produce *different* tables
for the same seed, so text watermarked on a Mac (MPS) would not verify on CPU with the same
keys. ``PortableSynthIDConfig`` always draws the table on CPU and then moves it to the model's
device, so the watermark depends only on the config, never on the hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SynthIDTextWatermarkingConfig,
    SynthIDTextWatermarkLogitsProcessor,
)

from .config import Config, GenerationConfig, ModelConfig, WatermarkConfig


# --------------------------------------------------------------------------- processor


def build_processor(wm: WatermarkConfig, device: str | torch.device = "cpu") -> SynthIDTextWatermarkLogitsProcessor:
    """transformers' SynthID logits processor with a device-independent sampling table."""
    proc = SynthIDTextWatermarkLogitsProcessor(
        ngram_len=wm.ngram_len,
        keys=list(wm.keys),
        sampling_table_size=wm.sampling_table_size,
        sampling_table_seed=wm.sampling_table_seed,
        context_history_size=wm.context_history_size,
        device=torch.device("cpu"),
        skip_first_ngram_calls=wm.skip_first_ngram_calls,
    )
    device = torch.device(device)
    if device.type != "cpu":
        proc.sampling_table = proc.sampling_table.to(device)
        proc.keys = proc.keys.to(device)
        proc.device = device
    return proc


class PortableSynthIDConfig(SynthIDTextWatermarkingConfig):
    """``SynthIDTextWatermarkingConfig`` whose processor uses a CPU-drawn sampling table.

    Passed to ``model.generate(watermarking_config=...)`` exactly like the base class, so the
    watermark still runs after temperature / top-k / top-p, where transformers places it.
    """

    @classmethod
    def from_watermark_config(cls, wm: WatermarkConfig) -> "PortableSynthIDConfig":
        return cls(
            ngram_len=wm.ngram_len,
            keys=list(wm.keys),
            context_history_size=wm.context_history_size,
            sampling_table_seed=wm.sampling_table_seed,
            sampling_table_size=wm.sampling_table_size,
            skip_first_ngram_calls=wm.skip_first_ngram_calls,
        )

    def _as_watermark_config(self) -> WatermarkConfig:
        return WatermarkConfig(
            keys=list(self.keys),
            ngram_len=self.ngram_len,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
        )

    def construct_processor(self, vocab_size: int, device) -> SynthIDTextWatermarkLogitsProcessor:
        proc = build_processor(self._as_watermark_config(), device)
        proc.debug_mode = self.debug_mode
        return proc


# --------------------------------------------------------------------------- devices / loading


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.float32 if device.type == "cpu" else torch.bfloat16
    try:
        return {"float32": torch.float32, "fp32": torch.float32, "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}[name]
    except KeyError:
        raise ValueError(f"unknown dtype {name!r}") from None


def load_tokenizer(model_id: str):
    """Tokenizer only (a few MB). Enough for verification; no weights are downloaded."""
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(mc: ModelConfig):
    """Model + tokenizer. Weights download lazily into the standard Hugging Face cache."""
    device = resolve_device(mc.device)
    dtype = resolve_dtype(mc.dtype, device)
    tok = load_tokenizer(mc.model_id)
    model = AutoModelForCausalLM.from_pretrained(mc.model_id, dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    return model, tok, device


# --------------------------------------------------------------------------- generation


@dataclass
class Generation:
    prompt: str
    text: str                 # decoded response (special tokens skipped)
    token_ids: list[int]      # response token ids exactly as sampled, up to and incl. EOS
    watermarked: bool


def format_prompt(tok, prompt: str, gen: GenerationConfig) -> str:
    if gen.use_chat_template and getattr(tok, "chat_template", None):
        kwargs = {}
        if not gen.enable_thinking:
            kwargs["enable_thinking"] = False  # Qwen3; other templates ignore unknown kwargs
        return tok.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, **kwargs
        )
    return prompt


class WatermarkedGenerator:
    """Holds a loaded model and generates with (or without) the SynthID watermark."""

    def __init__(self, cfg: Config, model=None, tokenizer=None, device=None):
        self.cfg = cfg
        if model is None:
            model, tokenizer, device = load_model(cfg.model)
        self.model, self.tokenizer, self.device = model, tokenizer, device

    @property
    def watermarking_config(self) -> PortableSynthIDConfig:
        return PortableSynthIDConfig.from_watermark_config(self.cfg.watermark)

    @torch.no_grad()
    def generate(self, prompts: str | list[str], watermark: bool = True) -> list[Generation]:
        if isinstance(prompts, str):
            prompts = [prompts]
        gen = self.cfg.generation
        if gen.seed is not None:
            torch.manual_seed(gen.seed)
        texts = [format_prompt(self.tokenizer, p, gen) for p in prompts]
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        kwargs = dict(
            max_new_tokens=gen.max_new_tokens,
            do_sample=gen.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if gen.do_sample:
            kwargs.update(temperature=gen.temperature, top_k=gen.top_k, top_p=gen.top_p)
        if watermark:
            kwargs["watermarking_config"] = self.watermarking_config
        out = self.model.generate(**enc, **kwargs)
        new = out[:, enc["input_ids"].shape[1]:].cpu()
        stop = eos_ids(self.tokenizer, self.model)
        results = []
        for prompt, row in zip(prompts, new.tolist()):
            ids = []
            for t in row:
                ids.append(t)
                if t in stop:
                    break
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            results.append(Generation(prompt=prompt, text=text, token_ids=ids, watermarked=watermark))
        return results


def eos_ids(tok, model=None) -> set[int]:
    """Every token id that ends a response (Qwen3 lists several: <|im_end|>, <|endoftext|>)."""
    ids: set[int] = set()
    for src in (tok.eos_token_id, tok.pad_token_id,
                getattr(getattr(model, "generation_config", None), "eos_token_id", None)):
        if src is None:
            continue
        ids.update(src if isinstance(src, (list, tuple)) else [src])
    return ids
