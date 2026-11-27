"""Shared fixtures: put promptset/scripts and synthid/ on sys.path; build a tiny random Qwen3 + tokenizer.

The tokenizer is the real Qwen3 one if ``/mnt/project-files/promptset_stage/qwen3_tokenizer`` (or
``$PROMPTSET_TEST_TOKENIZER``) exists; otherwise a ~1k-token byte-level BPE trained here on English +
Hindi text, with Qwen-style special tokens and a minimal chat template that honours ``enable_thinking``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
REPO = SCRIPTS.parent.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(REPO / "synthid"))

REAL_TOKENIZER = Path(os.environ.get("PROMPTSET_TEST_TOKENIZER", "/mnt/project-files/promptset_stage/qwen3_tokenizer"))

# transformers renders chat templates with trim_blocks/lstrip_blocks; no "-" whitespace control is
# used so that the literal newlines survive.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)

EN_TEXT = """The lighthouse keeper woke before dawn and climbed the narrow stairs. Rain had fallen all night,
and the sea was grey and restless. He trimmed the wick, wiped the glass, and watched the first boats leave
the harbour. "Another storm is coming," he said to nobody in particular. His daughter laughed at him!
Was he worried? Perhaps. The village depended on the light, and the light depended on him... Every
evening he wrote a short note in the log: wind, waves, ships, and the names of the fishermen who returned.
Students should consider whether technology helps or harms learning. Some argue that phones distract;
others believe that access to information makes classrooms more equal. In this essay I will explain both
views and give my own opinion, with examples from my school and from recent research."""

HI_TEXT = """सुबह होने से पहले ही दीपस्तंभ का रखवाला जाग गया और संकरी सीढ़ियाँ चढ़ने लगा। रात भर बारिश हुई थी,
और समुद्र धूसर और बेचैन था। उसने बाती ठीक की, शीशा पोंछा और बंदरगाह से निकलती पहली नावों को देखा।
"एक और तूफ़ान आ रहा है," उसने किसी से नहीं, बस यूँ ही कहा। उसकी बेटी हँस पड़ी! क्या वह चिंतित था? शायद।
गाँव रोशनी पर निर्भर था, और रोशनी उस पर॥ हर शाम वह लॉग में एक छोटा नोट लिखता था: हवा, लहरें, जहाज़,
और उन मछुआरों के नाम जो लौट आए। विद्यार्थियों को सोचना चाहिए कि तकनीक सीखने में मदद करती है या नुकसान।
कुछ लोग कहते हैं कि फ़ोन ध्यान भटकाते हैं; दूसरे मानते हैं कि जानकारी तक पहुँच कक्षाओं को अधिक बराबर बनाती है।"""


def _corpus() -> list[str]:
    texts = [EN_TEXT, HI_TEXT] * 20
    sam = REPO / "data" / "Samalochan"
    if sam.is_dir():
        for p in sorted(sam.glob("*.txt"))[:20]:
            texts.append(p.read_text(encoding="utf-8")[:20000])
    return texts


def build_tokenizer():
    if REAL_TOKENIZER.is_dir():
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(REAL_TOKENIZER))
        return tok, "qwen3"
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=1200, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tk.train_from_iterator(_corpus(), trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.add_tokens(["<think>", "</think>"])  # ordinary added tokens, as in Qwen3 (not skipped on decode)
    tok.chat_template = CHAT_TEMPLATE
    return tok, "stand-in-bpe"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A randomly initialised 2-layer Qwen3 saved with its tokenizer, loadable by from_pretrained(path)."""
    import torch
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM

    d = tmp_path_factory.mktemp("tiny_qwen3")
    tok, _ = build_tokenizer()
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    cfg = Qwen3Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      intermediate_size=128, vocab_size=len(tok), head_dim=16, max_position_embeddings=4096,
                      eos_token_id=[eos, pad], pad_token_id=pad, bos_token_id=pad, tie_word_embeddings=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    model.generation_config = GenerationConfig(eos_token_id=[eos, pad], pad_token_id=pad, bos_token_id=pad)
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return d
