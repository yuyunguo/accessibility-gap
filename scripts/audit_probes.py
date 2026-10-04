"""Adversarial audit of the linear-vs-nonlinear probe comparison.

Checks whether the conclusion "tuned linear ≈ tuned nonlinear" could be an
artifact of (i) leakage across texture families, (ii) an under-powered
nonlinear probe, (iii) boundary hyperparameters, or (iv) non-convergence.

Stimuli share generator seeds across α levels (stimulus k at every α is the
same texture family), so the honest generalization test groups all six
α-versions of a family into the same fold. All tuning uses inner CV on the
training fold only, grouped by family in grouped mode.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def family_index(labels: np.ndarray) -> np.ndarray:
    """Family id = position within its α level (files are concatenated level by level)."""
    fam = np.empty(len(labels), dtype=int)
    for a in np.unique(labels):
        idx = np.flatnonzero(labels == a)
        assert np.all(np.diff(idx) == 1), "labels must be level-contiguous"
        fam[idx] = np.arange(len(idx))
    return fam


def outer_splits(y: np.ndarray, fam: np.ndarray, mode: str, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    if mode == "grouped":
        return list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, y, fam))
    _, lev = np.unique(y, return_inverse=True)
    return list(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, lev))


def _scaled(est) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("est", est)])


def build_probes(dim: int, include_slow: bool, seed: int) -> Dict[str, Tuple[Pipeline, Dict]]:
    """Probe name -> (pipeline, hyperparameter grid). All are tuned by inner CV."""
    probes = {
        "ridge": (_scaled(Ridge()), {"est__alpha": list(np.logspace(-2, 6, 17))}),
        "kernel_rbf": (_scaled(KernelRidge(kernel="rbf")),
                       {"est__alpha": [1e-3, 1e-2, 1e-1, 1, 10],
                        "est__gamma": [g / dim for g in (0.03, 0.1, 0.3, 1.0, 3.0)]}),
        "knn": (_scaled(KNeighborsRegressor(weights="distance")), {"est__n_neighbors": [5, 10, 20, 40]}),
        "hgb": (HistGradientBoostingRegressor(random_state=seed, early_stopping=True),
                {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    }
    if include_slow:
        probes["mlp"] = (_scaled(MLPRegressor(hidden_layer_sizes=(128, 32), solver="lbfgs", max_iter=4000,
                                              random_state=seed)),
                         {"est__alpha": [0.1, 1.0, 10.0, 100.0]})
        probes["mlp_wide"] = (_scaled(MLPRegressor(hidden_layer_sizes=(512, 128), solver="lbfgs", max_iter=4000,
                                                   random_state=seed)),
                              {"est__alpha": [1.0, 10.0, 100.0]})
    return probes


def _boundary(grid: Dict, best: Dict) -> List[str]:
    """Names of hyperparameters whose selected value lies on the grid edge."""
    return [k for k, v in grid.items() if len(v) > 1 and best[k] in (v[0], v[-1])]


def run_mode(X: np.ndarray, y: np.ndarray, fam: np.ndarray, mode: str, probes: Dict, n_splits: int,
             seed: int, n_jobs: int) -> Dict:
    res: Dict[str, Dict[str, list]] = {n: {"test": [], "train": [], "best": [], "boundary": [], "n_iter": []}
                                       for n in probes}
    res["ridge_permuted"] = {"test": []}
    res["kernel_rbf_permuted"] = {"test": []}
    rng = np.random.default_rng(seed)
    for k, (tr, te) in enumerate(outer_splits(y, fam, mode, n_splits, seed)):
        inner = GroupKFold(3, shuffle=True, random_state=seed) if mode == "grouped" else 3
        fit_kw = {"groups": fam[tr]} if mode == "grouped" else {}
        for name, (pipe, grid) in probes.items():
            gs = GridSearchCV(clone(pipe), grid, cv=inner, n_jobs=n_jobs).fit(X[tr], y[tr], **fit_kw)
            est = gs.best_estimator_
            res[name]["test"].append(float(est.score(X[te], y[te])))
            res[name]["train"].append(float(est.score(X[tr], y[tr])))
            res[name]["best"].append({p: float(v) for p, v in gs.best_params_.items()})
            res[name]["boundary"].append(_boundary(grid, gs.best_params_))
            mlp = est.named_steps.get("est") if isinstance(est, Pipeline) else None
            if isinstance(mlp, MLPRegressor):
                res[name]["n_iter"].append(int(mlp.n_iter_))
        # Permutation null: shuffle labels within the training fold, keep test labels intact.
        y_perm = rng.permutation(y[tr])
        for name in ("ridge", "kernel_rbf"):
            pipe, grid = probes[name]
            gs = GridSearchCV(clone(pipe), grid, cv=inner, n_jobs=n_jobs).fit(X[tr], y_perm, **fit_kw)
            res[f"{name}_permuted"]["test"].append(float(gs.best_estimator_.score(X[te], y[te])))
        logger.info(f"  [{mode}] fold {k + 1}/{n_splits}: " + " ".join(
            f"{n}={res[n]['test'][-1]:+.3f}" for n in res if res[n]["test"]))
    return res


def summarize(res: Dict) -> Dict[str, float]:
    return {n: float(np.mean(r["test"])) for n, r in res.items() if r["test"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", nargs="+", type=Path)
    parser.add_argument("--slow", nargs="*", default=[], help="file stems that also get MLP probes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=14)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    out: Dict[str, Dict] = {}
    for f in args.features:
        d = np.load(f)
        X, y = d["features"].astype(np.float64), d["labels"].astype(np.float64)
        fam = family_index(y)
        probes = build_probes(X.shape[1], include_slow=f.stem in args.slow, seed=args.seed)
        out[f.stem] = {}
        for mode in ("random", "grouped"):
            t0 = time.time()
            r = run_mode(X, y, fam, mode, probes, args.n_splits, args.seed, args.n_jobs)
            s = summarize(r)
            out[f.stem][mode] = {"summary": s, "detail": r}
            logger.info(f"RESULT {f.stem} [{mode}] ({time.time() - t0:.0f}s): " +
                        " ".join(f"{k}={v:+.3f}" for k, v in s.items()))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
