# accessibility-gap

Code for probing the linear accessibility of a controlled fourth-order texture
statistic (α) in frozen representations of pretrained vision models.

## Layout

- `stimuli/` — generation of binary maximum-entropy (BTC) texture stimuli via MATLAB.
- `legacy/` — original analysis scripts, kept unchanged except for path cleanup,
  for reproducing previously reported values (reproduction baseline).

New revision code (seeded feature extraction, repeated stratified CV, paired
hierarchical bootstrap, diffusion controls) will be added under `src/` and
`scripts/` with Hydra configs in `run/conf/`.

## Requirements

Python 3.12; see `requirements.txt`. A CUDA GPU is needed for feature extraction.

## Stimuli

Stimulus generation requires MATLAB and the BTC maximum-entropy texture toolbox,
which is not distributed in this repository. Set:

- `MATLAB_BIN` — path to the MATLAB executable (default: `matlab`)
- `BTC_CODE_DIR` — path to the BTC toolbox root (default: `jv`)
- `STIM_DIR` — output directory for `.mat` stimuli (default: `data/stimuli`)

Parameter order in `paramvals` is `[g, b, c, d, e, t, u, v, w, a]`; the
fourth-order parameter α is the last entry.

## Data and results

Stimuli, cached features, and results are not versioned. Scripts read from
`data/` and write to `results/` relative to the repository root.
