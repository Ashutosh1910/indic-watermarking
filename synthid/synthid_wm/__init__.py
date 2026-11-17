"""SynthID-Text watermarking (generation + verification) on top of Hugging Face transformers."""

from .config import Config, DetectionConfig, GenerationConfig, ModelConfig, WatermarkConfig, load_config
from .verify import SynthIDVerifier, VerifyResult
from .watermark import (
    Generation,
    PortableSynthIDConfig,
    WatermarkedGenerator,
    build_processor,
    load_model,
    load_tokenizer,
    resolve_device,
)

__all__ = [
    "Config", "DetectionConfig", "GenerationConfig", "ModelConfig", "WatermarkConfig", "load_config",
    "SynthIDVerifier", "VerifyResult",
    "Generation", "PortableSynthIDConfig", "WatermarkedGenerator", "build_processor", "load_model",
    "load_tokenizer", "resolve_device",
]
