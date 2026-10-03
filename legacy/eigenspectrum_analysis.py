"""
Eigenspectrum Analysis (EXP-D)
================================
Addresses reviewer concern: "R² differences may reflect dimensional anisotropy."

Computes participation ratio (PR) and eigenspectrum statistics for each model's
feature matrix on the BTC stimuli. PR = (Σλᵢ)² / Σλᵢ² measures effective
dimensionality; anisotropic representations (low PR) concentrate variance in
few directions, which could artificially inflate or deflate probe R².

Models: all cached feature files in results/revision/feature_cache/
Output: results/revision/eigenspectrum_results.json

Usage
-----
  python scripts/eigenspectrum_analysis.py
"""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR   = Path(__file__).parent
CACHE_DIR    = SCRIPT_DIR.parent / "results" / "revision" / "feature_cache"
RESULTS_DIR  = SCRIPT_DIR.parent / "results" / "revision"

MODEL_LABELS = {
    "clip_features.npz":           "Contrastive (CLIP)",
    "mae_features.npz":            "Reconstructive (MAE)",
    "resnet_features.npz":         "Classification (ResNet50)",
    "sd_unet_features.npz":        "Diffusion (SD UNet)",
    "dino_features.npz":           "DINO (ViT-B/16)",
    "supervised_vit_features.npz": "Supervised ViT-B/16",
}


def participation_ratio(eigenvalues: np.ndarray) -> float:
    lam = eigenvalues[eigenvalues > 0]
    return float(np.sum(lam) ** 2 / np.sum(lam ** 2))


def analyse_model(features: np.ndarray, name: str) -> dict:
    N, D = features.shape
    sc = StandardScaler()
    X = sc.fit_transform(features)

    # Full covariance eigenspectrum via PCA
    n_comp = min(N - 1, D)
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(X)
    ev = pca.explained_variance_

    pr = participation_ratio(ev)
    pr_normalised = pr / D  # fraction of dimensions that are "effective"

    # Top-k cumulative variance
    cumvar = np.cumsum(ev) / np.sum(ev)
    k50 = int(np.searchsorted(cumvar, 0.50)) + 1
    k90 = int(np.searchsorted(cumvar, 0.90)) + 1
    k99 = int(np.searchsorted(cumvar, 0.99)) + 1

    logger.info(
        f"  {name}: D={D}, N={N}, PR={pr:.1f} ({pr_normalised*100:.1f}% of D), "
        f"dims for 50/90/99% var: {k50}/{k90}/{k99}"
    )

    return {
        "D": D,
        "N": N,
        "participation_ratio": pr,
        "pr_fraction": pr_normalised,
        "dims_50pct_var": k50,
        "dims_90pct_var": k90,
        "dims_99pct_var": k99,
        "top10_var_fraction": float(np.sum(ev[:10]) / np.sum(ev)),
        "top1_var_fraction":  float(ev[0] / np.sum(ev)),
        "explained_variance_ratio_top64": ev[:64].tolist(),
    }


def main():
    results = {}
    for fname, label in MODEL_LABELS.items():
        cache_path = CACHE_DIR / fname
        if not cache_path.exists():
            logger.warning(f"Cache not found: {cache_path}")
            continue
        logger.info(f"\n  {label}")
        data = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        results[label] = analyse_model(features, label)

    out_path = RESULTS_DIR / "eigenspectrum_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    logger.info("\n" + "="*70)
    logger.info("  EIGENSPECTRUM SUMMARY")
    logger.info("="*70)
    logger.info(f"  {'Model':<35} {'D':>6} {'PR':>7} {'PR/D%':>7} {'dim@90%':>8}")
    logger.info("  " + "-"*65)
    for label, r in results.items():
        logger.info(
            f"  {label:<35} {r['D']:>6} {r['participation_ratio']:>7.1f} "
            f"{r['pr_fraction']*100:>6.1f}% {r['dims_90pct_var']:>8}"
        )


if __name__ == "__main__":
    main()
