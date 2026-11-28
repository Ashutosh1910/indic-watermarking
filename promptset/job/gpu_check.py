"""GPU check for a new server: load the study model, generate a short watermarked answer in English and Hindi,
verify it on CPU, and report speed and memory. Uses the study's run config (model, sampling) with the public
demo keys, so it never touches the private key file.

    python promptset/job/gpu_check.py [--config promptset/config/generation.yaml] [--tokens 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "promptset" / "scripts"))
import generate as G  # noqa: E402
from _common import add_synthid_to_path  # noqa: E402

add_synthid_to_path()
import torch  # noqa: E402
from synthid_wm import SynthIDVerifier, WatermarkedGenerator  # noqa: E402

PROMPTS = {"en": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
           "hi": "एक दीपस्तंभ के रखवाले के बारे में एक छोटी कहानी लिखिए, जिसे एक बोतल में संदेश मिलता है।"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO / "promptset" / "config" / "generation.yaml"))
    ap.add_argument("--tokens", type=int, default=200)
    args = ap.parse_args()
    cfg, _ = G.load_run_config(args.config)
    cfg.generation.max_new_tokens = args.tokens
    if not torch.cuda.is_available():
        print("FAIL: no CUDA GPU visible (torch", torch.__version__, ")")
        return 1
    print(f"torch {torch.__version__} cuda {torch.version.cuda} gpu {torch.cuda.get_device_name(0)}")
    t0 = time.perf_counter()
    gen = WatermarkedGenerator(cfg)
    print(f"model {cfg.model.model_id} loaded in {time.perf_counter() - t0:.0f} s on {gen.device}")
    ver = SynthIDVerifier(cfg.watermark, gen.tokenizer, cfg.detection)
    ok = True
    for lang, prompt in PROMPTS.items():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = gen.generate([prompt], watermark=True)[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        r = ver.verify_text(out.text)
        print(f"{lang}: {len(out.token_ids)} tokens in {dt:.1f} s ({len(out.token_ids) / dt:.1f} tok/s, batch 1), "
              f"z = {r.z:.2f}, n_scored = {r.n_scored}, decision {r.decision}")
        print("   ", out.text[:160].replace("\n", " "), "...")
        # English must show the watermark at this length; Hindi is reported only (its per-token signal is weaker)
        ok &= len(out.token_ids) > 50 and (lang != "en" or r.z > 2.0)
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("GPU check:", "OK" if ok else "FAIL (English z <= 2 or a very short answer: look at the texts above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
