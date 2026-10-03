"""
Probe Robustness Analysis (EXP-B)
===================================
Addresses reviewer concern: "The effect depends on probe tuning."

Demonstrates that the accessibility gap is stable across a wide range of
MLP and probe hyperparameters for the two key models:
  - SD UNet (large gap)
  - MAE (small gap)

Varied parameters:
  - MLP hidden width:  {64, 128, 256, 512}
  - MLP depth:         {1, 2, 3} layers
  - PCA dimension:     {32, 64, 128, 256, full (no PCA)}
  - Ridge lambda:      {0.001, 0.01, 0.1, 1.0, 10.0}
  - Random seeds:      5 per configuration

Output: results/revision/probe_robustness_results.json

Usage
-----
  python scripts/probe_robustness.py
"""

import json
import logging
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR   = Path(__file__).parent
CACHE_DIR    = SCRIPT_DIR.parent / "results" / "revision" / "feature_cache"
RESULTS_DIR  = SCRIPT_DIR.parent / "results" / "revision"

SEEDS = [42, 123, 456, 789, 1024]

# pca_dim=None (full-dim MLP) is intentionally excluded: it is already the primary
# result in EXP-1 / Table 1. The robustness sweep targets the MLP-PCA family only,
# where lbfgs is fast (small projected d). The linear baseline always uses the
# full-dimensional Ridge to match the main analysis definition of delta.
SWEEP_CONFIG = {
    "hidden_width": [64, 128, 256, 512],
    "depth":        [1, 2, 3],
    "pca_dim":      [32, 64, 128, 256],          # MLP PCA dim only (no full-dim)
    "ridge_lambda": [0.001, 0.01, 0.1, 1.0, 10.0],
}

MODELS = {
    "SD UNet (Diffusion)": "sd_unet_features.npz",
    "MAE (Reconstructive)": "mae_features.npz",
}


# ---------------------------------------------------------------------------
# Single-config probe
# ---------------------------------------------------------------------------

def run_config_precomputed(
    X_tr_pca: np.ndarray,
    X_te_pca: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    hidden_width: int,
    depth: int,
    ridge_lambda: float,
    seed: int,
    r2_lin: float,
) -> Dict:
    """Fit MLP on already-projected features. Linear baseline is precomputed."""
    hidden = tuple([hidden_width] * depth)
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden, activation="relu", solver="adam",
        alpha=ridge_lambda, max_iter=300, random_state=seed, n_iter_no_change=20,
    )
    mlp.fit(X_tr_pca, y_tr)
    r2_mlp = float(mlp.score(X_te_pca, y_te))
    return {"r2_linear": r2_lin, "r2_mlp": r2_mlp, "delta": r2_mlp - r2_lin}


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep_model(
    features: np.ndarray,
    labels: np.ndarray,
    model_name: str,
) -> Dict:
    """
    Run the full hyperparameter sweep for one model.
    Returns dict with per-config and per-axis summary statistics.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Precompute full-dim linear baseline once (alpha=1.0 fixed)
    sc_base = StandardScaler().fit(X_tr)
    X_tr_sc = sc_base.transform(X_tr)
    X_te_sc = sc_base.transform(X_te)
    pipe_lin_base = Ridge(alpha=1.0).fit(X_tr_sc, y_tr)
    r2_lin_base = float(pipe_lin_base.score(X_te_sc, y_te))
    logger.info(f"  Full-dim linear baseline R² = {r2_lin_base:+.4f} (precomputed, fixed)")

    # Precompute PCA projections for each requested dimension — the expensive part
    logger.info("  Precomputing PCA projections...")
    pca_cache: Dict[int, tuple] = {}
    for pca_dim in SWEEP_CONFIG["pca_dim"]:
        n_comp = min(pca_dim, X_tr_sc.shape[1], X_tr_sc.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=42).fit(X_tr_sc)
        Xtr_p = pca.transform(X_tr_sc)
        Xte_p = pca.transform(X_te_sc)
        pca_cache[pca_dim] = (Xtr_p, Xte_p)
        logger.info(f"    PCA-{n_comp}: done")

    configs = list(itertools.product(
        SWEEP_CONFIG["hidden_width"],
        SWEEP_CONFIG["depth"],
        SWEEP_CONFIG["pca_dim"],
        SWEEP_CONFIG["ridge_lambda"],
        SEEDS,
    ))
    logger.info(f"  {len(configs)} MLP-only configurations")

    all_rows = []
    for hw, depth, pca_dim, ridge_lambda, seed in tqdm(configs, desc=f"  {model_name}"):
        Xtr_p, Xte_p = pca_cache[pca_dim]
        res = run_config_precomputed(Xtr_p, Xte_p, y_tr, y_te,
                                     hidden_width=hw, depth=depth,
                                     ridge_lambda=ridge_lambda, seed=seed,
                                     r2_lin=r2_lin_base)
        all_rows.append({
            "hidden_width": hw,
            "depth": depth,
            "pca_dim": pca_dim if pca_dim is not None else "full",
            "ridge_lambda": ridge_lambda,
            "seed": seed,
            **res,
        })

    deltas = np.array([r["delta"] for r in all_rows])
    r2_lins = np.array([r["r2_linear"] for r in all_rows])
    r2_mlps = np.array([r["r2_mlp"] for r in all_rows])

    # Summaries collapsed over seeds (mean across 5 seeds per config)
    def _by_axis(key, values_key):
        unique_vals = sorted(set(r[key] for r in all_rows), key=lambda x: (str(x)))
        out = {}
        for v in unique_vals:
            subset = [r[values_key] for r in all_rows if r[key] == v]
            out[str(v)] = {
                "mean": float(np.mean(subset)),
                "std":  float(np.std(subset)),
                "min":  float(np.min(subset)),
                "max":  float(np.max(subset)),
            }
        return out

    summary = {
        "n_configs": len(configs),
        "delta_overall": {
            "mean": float(np.mean(deltas)),
            "std":  float(np.std(deltas)),
            "min":  float(np.min(deltas)),
            "max":  float(np.max(deltas)),
            "pct5":  float(np.percentile(deltas, 5)),
            "pct95": float(np.percentile(deltas, 95)),
        },
        "r2_linear_overall": {
            "mean": float(np.mean(r2_lins)),
            "std":  float(np.std(r2_lins)),
        },
        "r2_mlp_overall": {
            "mean": float(np.mean(r2_mlps)),
            "std":  float(np.std(r2_mlps)),
        },
        "by_hidden_width":  _by_axis("hidden_width", "delta"),
        "by_depth":         _by_axis("depth", "delta"),
        "by_pca_dim":       _by_axis("pca_dim", "delta"),
        "by_ridge_lambda":  _by_axis("ridge_lambda", "delta"),
    }

    return {"summary": summary, "all_configs": all_rows}


def main():
    results = {}
    for model_name, cache_fname in MODELS.items():
        cache_path = CACHE_DIR / cache_fname
        if not cache_path.exists():
            logger.warning(f"Cache not found for {model_name}: {cache_path}")
            continue
        logger.info(f"\n{'='*60}")
        logger.info(f"  {model_name}")
        logger.info(f"{'='*60}")
        data     = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        labels   = data["labels"]
        logger.info(f"  D={features.shape[1]}, N={len(labels)}")
        results[model_name] = sweep_model(features, labels, model_name)

        s = results[model_name]["summary"]["delta_overall"]
        logger.info(
            f"  ΔR² overall: mean={s['mean']:+.4f} ± {s['std']:.4f}  "
            f"range=[{s['min']:+.4f}, {s['max']:+.4f}]  "
            f"5–95th pct=[{s['pct5']:+.4f}, {s['pct95']:+.4f}]"
        )

    out_path = RESULTS_DIR / "probe_robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # Quick human-readable summary
    logger.info("\n" + "="*60)
    logger.info("  ROBUSTNESS SUMMARY")
    logger.info("="*60)
    for model_name, res in results.items():
        s = res["summary"]["delta_overall"]
        logger.info(f"\n  {model_name}")
        logger.info(f"    ΔR²  mean ± std = {s['mean']:+.4f} ± {s['std']:.4f}")
        logger.info(f"    ΔR²  5th–95th   = [{s['pct5']:+.4f}, {s['pct95']:+.4f}]")
        logger.info(f"    ΔR²  min / max  = {s['min']:+.4f} / {s['max']:+.4f}")


if __name__ == "__main__":
    main()
