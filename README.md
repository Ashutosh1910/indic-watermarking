# indic-watermarking

Experiments on SynthID-Text watermark detection, including Hindi and Tamil.

- [`retokenise-experiment/`](retokenise-experiment/): how much of a watermark survives a plain decode → re-encode round trip.
- [`corpora/`](corpora/): one loader for every corpus under `data/` (`import corpora; corpora.get("samalochan")`). Adding a dataset takes one registration.
- [`synthid/`](synthid/): generate SynthID-watermarked text with any Hugging Face model (default Qwen3-8B) and verify it; library + `python -m demo` CLI.
- [`promptset/`](promptset/): the study (English vs Hindi, editing level vs length, Qwen3-14B). To run it on a GPU server: [`promptset/JOB.md`](promptset/JOB.md) (setup script, one command, analysis).
