"""End-to-end smoke test with a real model (default Qwen/Qwen3-0.6B, ~1.2 GB on first run).

Checks, on the same prompts:
  (a) watermarked on the accelerator, verified on CPU in this process with the tokenizer only  -> detected
  (b) the same texts with the wrong keys                                                     -> not detected
  (c) unwatermarked generations, and human-written Hindi passages from ../data              -> not detected
  (d) stock SynthIDTextWatermarkingConfig on MPS/CUDA, verified on CPU                       -> weak / lost
      (shows the device bug that PortableSynthIDConfig fixes; skipped on CPU-only machines)

    .venv/bin/python scripts/smoke_test.py [--model Qwen/Qwen3-0.6B] [--n 5] [--tokens 200]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import SynthIDTextWatermarkingConfig  # noqa: E402

from synthid_wm import SynthIDVerifier, WatermarkConfig, WatermarkedGenerator, load_config  # noqa: E402
from synthid_wm.watermark import eos_ids  # noqa: E402

PROMPTS = [
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain how monsoon rains form over the Indian subcontinent.",
    "Describe a busy railway station in the early morning.",
    "Give advice to a student starting their first research project.",
    "Write a letter to a friend about learning to cook.",
    "What are the pros and cons of living in a big city?",
    "Describe the history of the printing press in a few paragraphs.",
    "Write a poem in prose about the sea at night.",
]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--tokens", type=int, default=200)
ap.add_argument("--human", type=int, default=20, help="human-written passages to score")
args = ap.parse_args()

cfg = load_config(overrides={"model.model_id": args.model, "generation.max_new_tokens": args.tokens})
gen = WatermarkedGenerator(cfg)
print(f"model={args.model} device={gen.device} dtype={gen.model.dtype}")
prompts = PROMPTS[: args.n]
v = SynthIDVerifier(cfg.watermark, gen.tokenizer, cfg.detection, stop_ids=eos_ids(gen.tokenizer, gen.model))
wrong = SynthIDVerifier(WatermarkConfig(keys=[k + 7 for k in cfg.watermark.keys]), gen.tokenizer, cfg.detection)


def report(label, results):
    zs = [r.z for r in results]
    hits = sum(r.is_watermarked for r in results)
    print(f"{label:<44} detected {hits}/{len(results)}   z: " + " ".join(f"{z:6.1f}" for z in zs))
    return hits


wm = gen.generate(prompts, watermark=True)
plain = gen.generate(prompts, watermark=False)
ok = True
ok &= report("(a) watermarked, sampled ids", [v.verify_ids(g.token_ids) for g in wm]) == len(wm)
ok &= report("(a) watermarked, re-tokenised text", [v.verify_text(g.text) for g in wm]) == len(wm)
ok &= report("(b) watermarked text, wrong keys", [wrong.verify_text(g.text) for g in wm]) == 0
ok &= report("(c) unwatermarked text", [v.verify_text(g.text) for g in plain]) == 0

data = Path(__file__).resolve().parents[2] / "data" / "Hindi-Aesthetics-Corpus"
if data.is_dir():
    passages = []
    for f in sorted(data.iterdir()):
        text = f.read_text(encoding="utf-8", errors="ignore")
        ids = gen.tokenizer(text, add_special_tokens=False)["input_ids"]
        for i in range(0, len(ids) - args.tokens, args.tokens * 5):
            passages.append(gen.tokenizer.decode(ids[i:i + args.tokens]))
    passages = passages[: args.human]
    ok &= report(f"(c) human Hindi passages ({args.tokens} tok)", [v.verify_text(t) for t in passages]) <= 1

if gen.device.type != "cpu":
    stock = SynthIDTextWatermarkingConfig(**{k: getattr(cfg.watermark, k) for k in (
        "keys", "ngram_len", "sampling_table_size", "sampling_table_seed", "context_history_size")})
    enc = gen.tokenizer([gen.tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for p in prompts], return_tensors="pt", padding=True).to(gen.device)
    outs = gen.model.generate(**enc, watermarking_config=stock, do_sample=True, top_k=40, temperature=0.7,
                              top_p=1.0, max_new_tokens=args.tokens, pad_token_id=gen.tokenizer.pad_token_id)
    texts = gen.tokenizer.batch_decode(outs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    report(f"(d) stock config on {gen.device.type}, verified on CPU", [v.verify_text(t) for t in texts])

print("SMOKE TEST", "PASSED" if ok else "FAILED")
sys.exit(0 if ok else 1)
