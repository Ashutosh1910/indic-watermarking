# indic-watermarking

Experiments on SynthID-Text watermark detection, including Hindi and Tamil.

- [`retokenise-experiment/`](retokenise-experiment/): how much of a watermark survives a plain decode → re-encode round trip.
- [`synthid/`](synthid/): generate SynthID-watermarked text with any Hugging Face model (default Qwen3-8B) and verify it; library + `python -m demo` CLI.
