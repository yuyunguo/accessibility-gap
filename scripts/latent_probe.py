"""Noised-latent control: is α present in the UNet's input z_t?

Rebuilds z_t = add_noise(z0, ε, t) exactly as extract_sd_seeded.py does (same
scheduler, same seeded ε over the full latent tensor), then probes α from
  * full:   all 4×64×64 latent values (upper bound on linearly available information)
  * pooled: the 4 channel means (the same spatial pooling applied to UNet features)
with the tuned ridge and tuned RBF kernel probes used in audit_probes.py.
t = 0 means the clean latent z0.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from diffusers import PNDMScheduler
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from audit_probes import build_probes

logger = logging.getLogger(__name__)


def noised(z0: torch.Tensor, seed: int, t: int, scheduler) -> torch.Tensor:
    """Exact reconstruction of the UNet input used in extract_sd_seeded.py."""
    if t == 0:
        return z0
    eps = torch.randn(z0.shape, generator=torch.Generator(device="cpu").manual_seed(seed))
    tt = torch.full((len(z0),), t, dtype=torch.long)
    return scheduler.add_noise(z0, eps, tt)


def probe(X: np.ndarray, y: np.ndarray, n_jobs: int) -> Dict[str, float]:
    """Tuned ridge and RBF kernel ridge; shuffled stratified inner and outer CV."""
    probes = {k: v for k, v in build_probes(X.shape[1], include_slow=False, seed=0).items()
              if k in ("ridge", "kernel_rbf")}
    lev = np.unique(y, return_inverse=True)[1]
    out = {k: [] for k in probes}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, lev):
        inner = list(StratifiedKFold(3, shuffle=True, random_state=0).split(tr, lev[tr]))
        for name, (pipe, grid) in probes.items():
            gs = GridSearchCV(clone(pipe), grid, cv=inner, n_jobs=n_jobs).fit(X[tr], y[tr])
            out[name].append(float(gs.best_estimator_.score(X[te], y[te])))
    return {k: float(np.mean(v)) for k, v in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--latents", type=Path, required=True, help="vae_latents_*.npz from extract_sd_seeded.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    d = np.load(args.latents)
    z0, y = torch.from_numpy(d["latents"]), d["labels"].astype(np.float64)
    scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    conditions = [(None, 0)] + [(s, t) for s in args.seeds for t in args.timesteps]
    results = []
    for seed, t in conditions:
        zt = noised(z0, seed if seed is not None else 0, t, scheduler).numpy().astype(np.float64)
        for rep, X in (("full", zt.reshape(len(zt), -1)), ("pooled", zt.mean(axis=(2, 3)))):
            r = probe(X, y, args.n_jobs)
            row = {"latents": args.latents.name, "seed": seed, "t": t, "rep": rep, "dim": X.shape[1], **r}
            results.append(row)
            logger.info(f"RESULT {args.latents.stem} t={t} seed={seed} {rep:6s} (d={X.shape[1]}): "
                        f"ridge={r['ridge']:+.3f} kernel_rbf={r['kernel_rbf']:+.3f} gap={r['kernel_rbf'] - r['ridge']:+.3f}")
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
