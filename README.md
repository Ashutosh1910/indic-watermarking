# indic-watermarking

Experiments on SynthID-Text watermark detection, including Hindi and Tamil.

- [`retokenise-experiment/`](retokenise-experiment/): how much of a watermark survives a plain decode → re-encode round trip.
- [`corpora/`](corpora/): one loader for every corpus under `data/` (`import corpora; corpora.get("samalochan")`). Adding a dataset takes one registration.
