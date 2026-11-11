"""Round-2 Phase 1, item 5: leave-one-α-level-out generalization.

For each of the six nominal α levels, ridge and RBF kernel ridge (same pipelines and
grids as final_probe.py) are tuned by shuffled, stratified 3-fold inner CV on the other
five levels and predict every stimulus of the held-out level. Interior levels (0.1-0.4)
test interpolation; the end levels (0.05, 0.5) test extrapolation. Per level we report
MSE and bias (mean prediction − level); R² is reported only on the predictions pooled
over all six folds, because a single level has zero target variance.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from final_probe import CV_SEED, probes_for

logger = logging.getLogger(__name__)


def lolo(X: np.ndarray, y: np.ndarray, pipe, grid: Dict, n_jobs: int) -> Dict:
    """Held-out predictions for every level, plus per-level MSE/bias and pooled R²."""
    levels = np.unique(y)
    pred = np.full(len(y), np.nan)
    per_level, chosen = {}, {}
    for a in levels:
        tr, te = np.flatnonzero(y != a), np.flatnonzero(y == a)
        lev = np.searchsorted(levels, y[tr])
        inner = list(StratifiedKFold(3, shuffle=True, random_state=CV_SEED).split(tr, lev))
        gs = GridSearchCV(clone(pipe), grid, cv=inner, n_jobs=n_jobs).fit(X[tr], y[tr])
        pred[te] = gs.best_estimator_.predict(X[te])
        err = pred[te] - a
        per_level[f"{a:g}"] = {"mse": float(np.mean(err ** 2)), "bias": float(err.mean()),
                               "pred_sd": float(pred[te].std(ddof=1))}
        chosen[f"{a:g}"] = {p: float(v) for p, v in gs.best_params_.items()}
    assert not np.isnan(pred).any()
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    interior = (y > levels[0]) & (y < levels[-1])
    r2_int = 1 - np.sum((y[interior] - pred[interior]) ** 2) / np.sum((y[interior] - y[interior].mean()) ** 2)
    return {"r2_pooled": float(r2), "r2_interior_pooled": float(r2_int), "per_level": per_level, "chosen": chosen}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    for f in args.features:
        if f.stem in out:
            logger.info("skip %s (done)", f.stem)
            continue
        d = np.load(f)
        X, y = d["features"].astype(np.float64), d["labels"].astype(np.float64)
        res = {name: lolo(X, y, pipe, grid, args.n_jobs) for name, (pipe, grid) in probes_for(X.shape[1], False).items()}
        res["gap_pooled"] = res["kernel_rbf"]["r2_pooled"] - res["ridge"]["r2_pooled"]
        out[f.stem] = res
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))      # checkpoint after every file
        logger.info("%s: ridge %.4f kernel %.4f (pooled LOLO R²)", f.stem, res["ridge"]["r2_pooled"],
                    res["kernel_rbf"]["r2_pooled"])


if __name__ == "__main__":
    main()
