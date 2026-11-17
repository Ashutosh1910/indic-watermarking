"""Shared settings."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Keep downloaded tokenizer/dataset files inside this folder unless the user already set HF_HOME.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

# Only the tokenizer is downloaded (a few MB). Every Qwen3 size shares it, including Qwen3-8B.
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
# SynthID-Text ngram_len: a g-value at position t depends on tokens t-4..t.
NGRAM_LEN = 5
