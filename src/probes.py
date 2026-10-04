"""Linear and nonlinear probes with matched, nested hyperparameter tuning.

Both probes see identical standardized, full-dimensional inputs. Regularization
strength is selected by inner cross-validation on the training fold only, so the
linear-vs-nonlinear comparison is not driven by a fixed, mismatched penalty.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeConfig:
    """Hyperparameters for the probe family."""

    # Grids extended (2026-10-04) after audit showed selections on edges: ridge hit 1e6 only
    # where R²≈0 (t=900); the MLP hit L2=1.0 (lowest) for high-R² models.
    ridge_alphas: Tuple[float, ...] = tuple(np.logspace(-2, 8, 41))
    mlp_hidden: Tuple[int, ...] = (128, 32)
    mlp_l2_grid: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    mlp_max_iter: int = 2000
    inner_folds: int = 3
    n_jobs: int = 8
    seed: int = 0


def stratified_splits(
    labels: np.ndarray, n_splits: int = 5, n_repeats: int = 10, seed: int = 0
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) stratified by discrete label level."""
    _, levels = np.unique(labels, return_inverse=True)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    yield from rskf.split(np.zeros(len(labels)), levels)


def _ridge_fixed(alpha: float) -> Pipeline:
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def _ridge_tuned(cfg: ProbeConfig) -> Pipeline:
    # Efficient leave-one-out (GCV) selection on the training fold only.
    return Pipeline([("scale", StandardScaler()), ("ridge", RidgeCV(alphas=cfg.ridge_alphas))])


def _mlp(cfg: ProbeConfig, l2: float, activation: str = "relu") -> Pipeline:
    mlp = MLPRegressor(
        hidden_layer_sizes=cfg.mlp_hidden, activation=activation, solver="lbfgs",
        alpha=l2, max_iter=cfg.mlp_max_iter, random_state=cfg.seed,
    )
    return Pipeline([("scale", StandardScaler()), ("mlp", mlp)])


def _mlp_tuned(cfg: ProbeConfig, activation: str = "relu") -> GridSearchCV:
    return GridSearchCV(
        _mlp(cfg, l2=cfg.mlp_l2_grid[0], activation=activation),
        {"mlp__alpha": list(cfg.mlp_l2_grid)},
        cv=KFold(cfg.inner_folds, shuffle=True, random_state=cfg.seed),
        n_jobs=cfg.n_jobs,
    )


def build_probes(cfg: ProbeConfig, include_mlp: bool = True) -> Dict[str, object]:
    """Return the named probe estimators evaluated on every fold."""
    probes: Dict[str, object] = {
        "ridge_fixed_l1": _ridge_fixed(1.0),          # legacy (submitted manuscript)
        "ridge_tuned": _ridge_tuned(cfg),
    }
    if include_mlp:
        probes["mlp_fixed_l0.1"] = _mlp(cfg, l2=0.1)  # legacy full-dim MLP
        probes["mlp_tuned"] = _mlp_tuned(cfg)
    return probes


def _selected_param(est: object) -> Optional[float]:
    if isinstance(est, GridSearchCV):
        return float(est.best_params_["mlp__alpha"])
    if isinstance(est, Pipeline) and isinstance(est[-1], RidgeCV):
        return float(est[-1].alpha_)
    return None


def evaluate_folds(
    X: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    probes: Dict[str, object],
) -> Dict[str, Dict[str, List[float]]]:
    """Fit every probe on every fold; record test R², train R², and selected penalty."""
    out: Dict[str, Dict[str, List[float]]] = {
        name: {"test_r2": [], "train_r2": [], "param": []} for name in probes
    }
    for k, (tr, te) in enumerate(splits):
        for name, proto in probes.items():
            est = clone(proto).fit(X[tr], y[tr])
            out[name]["test_r2"].append(float(est.score(X[te], y[te])))
            out[name]["train_r2"].append(float(est.score(X[tr], y[tr])))
            out[name]["param"].append(_selected_param(est))
        logger.info(f"  fold {k + 1}/{len(splits)} done")
    return out


def summarize(fold_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Mean and SD across folds for each probe, plus tuned and legacy gaps."""
    summ: Dict[str, Dict[str, float]] = {}
    for name, r in fold_results.items():
        t = np.asarray(r["test_r2"])
        summ[name] = {
            "test_mean": float(t.mean()), "test_sd": float(t.std(ddof=1)) if len(t) > 1 else 0.0,
            "train_mean": float(np.mean(r["train_r2"])), "n_folds": int(len(t)),
        }
    for gap, (mlp, lin) in {"gap_tuned": ("mlp_tuned", "ridge_tuned"),
                            "gap_legacy": ("mlp_fixed_l0.1", "ridge_fixed_l1")}.items():
        if mlp in fold_results and lin in fold_results:
            n = min(len(fold_results[mlp]["test_r2"]), len(fold_results[lin]["test_r2"]))
            d = np.asarray(fold_results[mlp]["test_r2"][:n]) - np.asarray(fold_results[lin]["test_r2"][:n])
            summ[gap] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)) if n > 1 else 0.0, "n_folds": n}
    return summ
