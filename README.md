# accessibility-gap

Code and result files for **"Linear Accessibility of a Fourth-Order Texture
Statistic across Vision Models: A Controlled Probe Analysis"** (Yu, submitted to
*Neural Networks*).

The study probes frozen features of seven vision models (CLIP, DINO, MAE,
ResNet50, supervised ViT-B/16, ImageGPT, Stable Diffusion v1.5) for a controlled
fourth-order texture statistic α, comparing ridge regression with RBF kernel
ridge (and a secondary MLP) under identical nested tuning, with a paired,
hierarchical bootstrap for uncertainty.

## Reproducing the reported numbers

Every number in the manuscript's Tables 2–5 and Figure 2 is in
`results/final/final_stats.json`, produced once by `run_final.sh` at the commit
tagged **`final-run`**. The figures can be regenerated from that file without a GPU:

```bash
pip install -r requirements.txt
python scripts/make_figure1.py --stats results/final/final_stats.json --out figures/fig2   # results figure
python scripts/verify_figure1.py results/final/final_stats.json                            # read-back check
```

`results/final/input_hashes.txt` lists the SHA-256 of every stimulus and feature
file that entered the final run.

## Full pipeline

1. **Stimuli** — `stimuli/generate_btc_octave.py` generates seeded BTC
   maximum-entropy textures with GNU Octave and writes a `manifest.json` (seed,
   parameters, validated realized statistics) per stimulus set. It requires the
   BTC texture toolbox of the Victor laboratory, which is openly available on
   figshare (doi:10.6084/m9.figshare.1409463) and is not redistributed here;
   point `BTC_CODE_DIR` (or `--btc-dir`) at it. The manifests of the stimulus
   sets used in the paper are in `data/stimuli_v2/*/manifest.json`.
2. **Features** — `scripts/extract_features_v2.py` (CLIP, DINO, MAE, ResNet50,
   ViT, ImageGPT; exact integer pixel replication) and
   `scripts/extract_sd_seeded.py` (Stable Diffusion v1.5, seeded noise).
   A CUDA GPU is needed.
3. **Final run** — `run_final.sh` runs the unit tests, the remaining extractions,
   the out-of-fold probes (`scripts/final_probe.py`), and the statistics
   (`scripts/final_stats.py`). Set `PYTHON` to your interpreter and
   `NATURAL_PATCHES` to the natural-texture patch file (built from the
   Describable Textures Dataset by `legacy/natural_texture_probe.py`).
4. **Supplementary analyses** — `scripts/audit_probes.py` (k-NN and
   gradient-boosting audit), `scripts/sd_block_sweep.py` (UNet stage sweep),
   `scripts/latent_probe.py` (noised-latent control),
   `scripts/make_supplement_tables.py` (prints the supplementary table cells).

## Layout

| Path | Contents |
|---|---|
| `src/probes.py` | Probe definitions and cross-validation utilities |
| `scripts/` | Extraction, final analysis, audits, figures, verifiers |
| `stimuli/` | Seeded Octave stimulus generator |
| `tests/` | Unit tests of splits, inner CV, replication, resampling, and statistics (`pytest -q tests/`) |
| `results/final/` | `final_stats.json`, `input_hashes.txt` |
| `data/stimuli_v2/` | Stimulus manifests (the `.mat` stimuli are not versioned) |
| `legacy/` | Scripts of an earlier version of this analysis, kept unchanged except for path cleanup so that its values can be reproduced (`scripts/reproduce_baseline.py`; see Supplementary Section S5 of the paper) |

BTC parameter order in `paramvals` is `[g, b, c, d, e, t, u, v, w, a]`; the
fourth-order parameter α is the last entry.

## License

MIT (see `LICENSE`). The BTC texture toolbox is distributed separately by its authors.
