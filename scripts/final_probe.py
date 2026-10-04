"""Final analysis, step 1: out-of-fold (OOF) predictions for every condition.

For one feature file (``features``, ``labels``), fits tuned ridge and tuned RBF
kernel ridge under 10 × stratified 5-fold CV and stores the OOF prediction of
every stimulus in every repeat. Optionally (``--mlp``) also a tuned MLP under one
repeat. Hyperparameters are tuned by shuffled, stratified 3-fold inner CV on the
training fold only. Statistics are computed separately (final_stats.py) from the
saved predictions, so no number depends on how folds were averaged.
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit_probes import build_probes

logger = logging.getLogger(__name__)

N_SPLITS, N_REPEATS, CV_SEED = 5, 10, 0
MLP_L2 = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def probes_for(dim: int, with_mlp: bool) -> Dict[str, Tuple[Pipeline, Dict]]:
    p = {k: v for k, v in build_probes(dim, include_slow=False, seed=0).items() if k in ("ridge", "kernel_rbf")}
    if with_mlp:
        mlp = MLPRegressor(hidden_layer_sizes=(128, 32), solver="lbfgs", max_iter=2000, random_state=0)
        p["mlp"] = (Pipeline([("scale", StandardScaler()), ("est", mlp)]), {"est__alpha": MLP_L2})
    return p


def strata(y: np.ndarray, max_levels: int = 20) -> np.ndarray:
    """Stratification labels: the α level for discrete targets, deciles for continuous ones."""
    u, inv = np.unique(y, return_inverse=True)
    if len(u) <= max_levels:
        return inv
    edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(edges, y, side="right")


def oof_predictions(X: np.ndarray, y: np.ndarray, pipe, grid: Dict, n_repeats: int, n_jobs: int) -> Dict:
    """OOF predictions (n_repeats × N), plus per-fold train R², chosen params, and grid-edge flags."""
    lev = strata(y)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=n_repeats, random_state=CV_SEED)
    oof = np.full((n_repeats, len(y)), np.nan)
    train_r2, chosen, edge = [], [], []
    for k, (tr, te) in enumerate(rskf.split(X, lev)):
        inner = list(StratifiedKFold(3, shuffle=True, random_state=CV_SEED).split(tr, lev[tr]))
        gs = GridSearchCV(clone(pipe), grid, cv=inner, n_jobs=n_jobs).fit(X[tr], y[tr])
        oof[k // N_SPLITS, te] = gs.best_estimator_.predict(X[te])
        train_r2.append(float(gs.best_estimator_.score(X[tr], y[tr])))
        chosen.append({p: float(v) for p, v in gs.best_params_.items()})
        edge.append([p for p, v in grid.items() if len(v) > 1 and gs.best_params_[p] in (v[0], v[-1])])
    assert not np.isnan(oof).any(), "every stimulus must be predicted exactly once per repeat"
    return {"oof": oof, "train_r2": train_r2, "chosen": chosen, "edge": edge}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path)
    parser.add_argument("--target-key", default="labels", help="npz key holding the regression target")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--mlp", action="store_true", help="also fit the tuned MLP (one repeat)")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    d = np.load(args.features)
    X, y = d["features"].astype(np.float64), d[args.target_key].astype(np.float64)
    assert X.ndim == 2 and len(X) == len(y) and np.isfinite(X).all(), "bad feature matrix"
    out = {"file": args.features.name, "sha256": sha256(args.features), "n": int(len(y)), "dim": int(X.shape[1]),
           "y": y, "cv": {"n_splits": N_SPLITS, "n_repeats": args.repeats, "seed": CV_SEED}}
    t0 = time.time()
    for name, (pipe, grid) in probes_for(X.shape[1], args.mlp).items():
        reps = 1 if name == "mlp" else args.repeats
        out[name] = oof_predictions(X, y, pipe, grid, reps, args.n_jobs)
        r2 = [1 - np.sum((y - o) ** 2) / np.sum((y - y.mean()) ** 2) for o in out[name]["oof"]]
        logger.info(f"{args.features.stem} {name}: OOF R² mean {np.mean(r2):+.4f} over {reps} repeat(s)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in out.items() if k not in ("y", "ridge", "kernel_rbf", "mlp")}
    meta |= {p: {k: v for k, v in out[p].items() if k != "oof"} for p in ("ridge", "kernel_rbf", "mlp") if p in out}
    np.savez(args.out_dir / f"oof_{args.features.stem}.npz", y=y,
             **{f"oof_{p}": out[p]["oof"] for p in ("ridge", "kernel_rbf", "mlp") if p in out})
    (args.out_dir / f"oof_{args.features.stem}.json").write_text(json.dumps(meta, indent=1))
    logger.info(f"DONE {args.features.stem} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
