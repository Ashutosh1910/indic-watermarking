# Analysis of a study dump (`scripts/analyze.py`)

The job runs this as its last stage. It reads a `run_study.py` dump and writes `<dump>/analysis/`. It is CPU
only, deterministic (seeded bootstrap), and never writes passage text.

```bash
python promptset/scripts/analyze.py dump/main --tokenizer Qwen/Qwen3-14B          # -> dump/main/analysis/
python promptset/scripts/analyze.py dump/main --set bootstrap.reps=200 --set primary.cutoff=800
```

```python
import sys; sys.path.insert(0, "promptset/scripts")
import analyze
res = analyze.run("dump/main", analyze.load_config(), tokenizer="Qwen/Qwen3-14B")   # tables, notes, figures
```

Settings: `config/analysis.yaml` (bootstrap reps and seed, primary comparison, f bins, z threshold).
A missing stream, human set or file is skipped and listed under "Notes" in `analysis.md`.

## Definitions

**Phases.** `human` = the human sets (`human.edit_anchors`, `human.pool`). `control` = the unwatermarked ctrl
stream. `edit` = L1-L4. `continue` = C_continue (k = 25/50/75 % human head). `ai_matched` = G_title and
O_outline (100 % AI, written from the same documents as the edits). `ai_prompt` = G (100 % AI from the prompt
set).

**Views.** `same_words` = passages cut at word cutoffs, `same_tokens` = cut at Qwen3 token cutoffs. Both come
from the same generations.

**Per-token strength** = z / sqrt(n_scored). For watermarked text z grows like sqrt(n_scored), so this is the
watermark signal per scored token, separate from how many tokens a passage has. `mask_share` = share of
positions the detector skipped because their 4-token context repeated.

**f** = share of the passage the model actually wrote. Human = 0; G, G_title, O_outline = 1; C rows use trim's
`ai_word_share` / `ai_token_share`. For L1-L4 the passage is aligned with its human input (`human_passage` of
the generation row) after NFC normalisation and whitespace collapse:

* `f_words`: share of the passage's words outside difflib matching blocks (autojunk off);
* `f_tokens`: share of the passage's tokens inside those non-copied words;
* `f_ngram`: share of the passage's token n-grams (n = the watermark's ngram_len) not found in the human
  input. This is the detector's view: a copied n-gram carries no watermark.

The alignment runs once per generation and is reused for the shorter (prefix) passages. Texts over 4,000 words
are aligned in windows (`f.max_align_words`). A 1,200-word Hindi passage (about 5,600 tokens) takes about
0.03 s, so a full study takes a few minutes.

**Clusters.** Every CI resamples `pair_id` clusters (one document, or one prompt pair for G): cutoffs of one
text are nested prefixes and the levels share documents, so passages are not independent. Ratio-of-sums
bootstrap, seeded.

**Primary comparison** (fixed before the run): Hindi minus English per-token strength per phase, in the
`same_tokens` view at 400 tokens (`primary` in analysis.yaml). It pairs on `pair_id` when both languages share
at least half of the smaller language's clusters (G), and resamples each language on its own otherwise.

## Outputs (`<dump>/analysis/`)

| file | what |
|---|---|
| `analysis.md` | primary result first, then every table and the caveats |
| `passages.csv` / `.jsonl` | one row per scored passage: phase, view, cutoff, z, n_scored, mask_share, strength, f_words, f_tokens, f_ngram, pair_id, metadata |
| `calibration.csv` | human and control text per language x view x cutoff: mean and sd of z, KS distance to N(0,1), FPR with Wilson CI, expected FPR, 0.999 quantile (n >= 1000), mask share |
| `strength.csv` | per stream x phase x language x level x k x view x cutoff: detection rate and mean strength with clustered CIs, mean f, mask share; `gb_*` columns weight genres equally |
| `lang_gap.csv` | hi - en strength with CI; `primary` marks the pre-registered rows |
| `f_curve.csv`, `f_fit.csv` | strength and detection by f_words bin (edit + continue); fit strength = c x f through the origin, against `c_matched_ai` (G_title / O_outline strength in the same cell) |
| `length_trap.csv` | tokens needed for mean z to reach the threshold, (thr / strength)^2 / (1 - mask_share), and the same in words |
| `retok.csv` | raw generations: z of the decoded text vs z of the sampled ids, and how often the flags disagree |
| `figures/*.png` | detection vs cutoff, strength vs f, human z histograms against N(0,1) (needs matplotlib) |
| `analysis_meta.json` | settings, counts, notes |

## Reading the results

* Judge calibration from the z distribution (mean near 0, sd near 1, small KS distance), not from flag counts:
  the expected FPR at z > 3 is 0.13 %, a handful of passages per cell.
* At equal words Hindi has about 4.5x the tokens, which raises z by about 2x on its own. Compare languages by
  strength, and at equal f.
* If `c_fit` is close to `c_matched_ai`, the signal of an edit is proportional to the share of new text it has.

## Caveats

* Hindi same-tokens passages are the opening of a text (1,500 tokens is about 330 Hindi words).
* Strength assumes a steady signal along the text; check with `load_dump.prefix_z`.
* f measures surface change. Report `f_words` and `f_ngram` together.
* The full study holds a few hundred MB in memory.

## Tests

```bash
python -m pytest promptset/tests/test_analyze.py -q -p no:cacheprovider
```
