"""Typed settings for SynthID-Text generation and verification, loaded from YAML.

One YAML file (see ``config/default.yaml``) has four blocks: ``model``, ``watermark``,
``generation`` and ``detection``. Every field has a default, so a file only needs the
fields it changes. ``load_config(path, overrides)`` applies dotted overrides such as
``{"watermark.ngram_len": 4}`` on top, which is how the demo CLI flags work.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

# Public demo keys from the Hugging Face SynthID-Text blog post (not secret).
BLOG_KEYS = [654, 400, 836, 123, 340, 443, 597, 160, 57, 29, 590, 639, 13, 715, 468,
             990, 966, 226, 324, 585, 118, 504, 421, 521, 129, 669, 732, 225, 90, 960]


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen3-8B"
    device: str = "auto"
    dtype: str = "auto"


@dataclass
class WatermarkConfig:
    """Exactly the arguments of transformers' ``SynthIDTextWatermarkingConfig``."""

    keys: list[int] = field(default_factory=lambda: list(BLOG_KEYS))
    ngram_len: int = 5
    sampling_table_size: int = 2**16
    sampling_table_seed: int = 0
    context_history_size: int = 1024
    skip_first_ngram_calls: bool = False

    def validate(self) -> None:
        if not self.keys:
            raise ValueError("watermark.keys must be a non-empty list of ints")
        if len(set(k % self.sampling_table_size for k in self.keys)) != len(self.keys):
            raise ValueError("watermark.keys must be distinct modulo sampling_table_size")
        if self.ngram_len < 2:
            raise ValueError("watermark.ngram_len must be >= 2")
        if not 1 <= self.sampling_table_size <= 2**24:
            raise ValueError("watermark.sampling_table_size must be in [1, 2**24]")

    @property
    def depth(self) -> int:
        return len(self.keys)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 1.0
    use_chat_template: bool = True
    enable_thinking: bool = False
    seed: int | None = 0


@dataclass
class DetectionConfig:
    method: str = "weighted_mean"  # "weighted_mean" | "mean"
    alpha: float = 0.01
    min_scored: int = 50
    null_mean: str | float = "table"
    bayesian_detector: str | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "model": ModelConfig,
    "watermark": WatermarkConfig,
    "generation": GenerationConfig,
    "detection": DetectionConfig,
}


def _build(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"unknown config section(s): {sorted(unknown)}")
    parts = {}
    for name, cls in _SECTIONS.items():
        values = raw.get(name) or {}
        known = {f.name for f in dataclasses.fields(cls)}
        bad = set(values) - known
        if bad:
            raise ValueError(f"unknown field(s) in '{name}': {sorted(bad)}")
        parts[name] = cls(**values)
    cfg = Config(**parts)
    cfg.watermark.keys = [int(k) for k in cfg.watermark.keys]
    cfg.watermark.validate()
    if cfg.detection.method not in ("weighted_mean", "mean"):
        raise ValueError("detection.method must be 'weighted_mean' or 'mean'")
    return cfg


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (default: config/default.yaml) and apply dotted overrides.

    ``overrides`` maps ``"section.field"`` to a value; ``None`` values are ignored so
    unset CLI flags fall through to the file.
    """
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw = yaml.safe_load(path.read_text()) or {}
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        section, _, name = dotted.partition(".")
        raw.setdefault(section, {})
        raw[section] = dict(raw[section] or {})
        raw[section][name] = value
    return _build(raw)
