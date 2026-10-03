"""
Revision Probing — Addressing Peer Review Issues
=================================================

Addresses:
  EXP-1  Full-dimensional MLP probe (no PCA) for all models — critical confound check
  EXP-2  Proper SSS for CLIP and ResNet50: Spearman rho between alpha and L2 distance
         from per-model mean-baseline (alpha=0) feature vector
  STAT-1 Paired bootstrap comparison: Delta-R² for SD UNet vs. every other paradigm
  INTERP-1 ImageGPT analysis: linear probe on PCA-64 features to confirm PCA causes
           negative gap; full-dim MLP to verify linear sufficiency of ImageGPT features

Usage
-----
  # All experiments (requires GPU for feature re-extraction):
  python scripts/revision_probing.py --device cuda

  # Probe only (reuse cached features):
  python scripts/revision_probing.py --probe-only --device cpu

  # Single model for quick test:
  python scripts/revision_probing.py --archs sd_unet --device cuda
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "revision"
FEATURE_CACHE_DIR = RESULTS_DIR / "feature_cache"

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

# ---------------------------------------------------------------------------
# Probe suite
# ---------------------------------------------------------------------------

def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def probe_all(
    features: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict:
    """
    Runs four probes on (features, labels):
      1. linear_full  : Ridge on full feature space
      2. linear_pca64 : Ridge on PCA-64 projected features
      3. mlp_pca64    : MLP on PCA-64 features (matches original paper protocol)
      4. mlp_full     : MLP on full feature space (EXP-1 critical control)

    Returns per-probe point estimates and bootstrap 95% CIs (paired resamples
    from the same held-out test set).
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    n_te = len(y_te)
    D = features.shape[1]
    n_pca = min(64, D, X_tr.shape[0] - 1)

    # ---- Probe 1: linear full ----
    pipe_lin_full = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe_lin_full.fit(X_tr, y_tr)
    r2_lin_full = float(pipe_lin_full.score(X_te, y_te))

    # ---- Probe 2: linear PCA-64 ----
    pipe_lin_pca = Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe_lin_pca.fit(X_tr, y_tr)
    r2_lin_pca = float(pipe_lin_pca.score(X_te, y_te))

    # ---- Probe 3: MLP PCA-64 (original protocol) ----
    pipe_mlp_pca = Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 32), activation="relu", solver="lbfgs",
            alpha=0.01, max_iter=2000, random_state=random_state,
        )),
    ])
    pipe_mlp_pca.fit(X_tr, y_tr)
    r2_mlp_pca = float(pipe_mlp_pca.score(X_te, y_te))

    # ---- Probe 4: MLP full-dimensional (EXP-1) ----
    # Use stronger regularization for high-d input; smaller hidden layers
    # to maintain roughly similar total parameter count relative to n_train.
    pipe_mlp_full = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 32), activation="relu", solver="lbfgs",
            alpha=0.1, max_iter=2000, random_state=random_state,
        )),
    ])
    pipe_mlp_full.fit(X_tr, y_tr)
    r2_mlp_full = float(pipe_mlp_full.score(X_te, y_te))

    # ---- Bootstrap: paired across all four probes ----
    rng = np.random.default_rng(random_state)
    boot = {k: [] for k in ["lin_full", "lin_pca", "mlp_pca", "mlp_full",
                             "delta_original", "delta_fulldim"]}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_te, size=n_te)
        Xb, yb = X_te[idx], y_te[idx]
        if len(np.unique(yb)) < 2:
            continue
        b_lf = float(pipe_lin_full.score(Xb, yb))
        b_lp = float(pipe_lin_pca.score(Xb, yb))
        b_mp = float(pipe_mlp_pca.score(Xb, yb))
        b_mf = float(pipe_mlp_full.score(Xb, yb))
        boot["lin_full"].append(b_lf)
        boot["lin_pca"].append(b_lp)
        boot["mlp_pca"].append(b_mp)
        boot["mlp_full"].append(b_mf)
        boot["delta_original"].append(b_mp - b_lf)   # original paper gap
        boot["delta_fulldim"].append(b_mf - b_lf)    # full-dim MLP vs. linear

    def _ci(samples):
        a = np.array(samples)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    # Spearman rho of linear-full predictions
    rho, pval = spearmanr(y_te, pipe_lin_full.predict(X_te))

    return {
        "linear_full":    {"r2": r2_lin_full,  "ci": _ci(boot["lin_full"])},
        "linear_pca64":   {"r2": r2_lin_pca,   "ci": _ci(boot["lin_pca"])},
        "mlp_pca64":      {"r2": r2_mlp_pca,   "ci": _ci(boot["mlp_pca"])},
        "mlp_full":       {"r2": r2_mlp_full,  "ci": _ci(boot["mlp_full"])},
        "delta_original": {
            "value": r2_mlp_pca - r2_lin_full,
            "ci": _ci(boot["delta_original"]),
        },
        "delta_fulldim": {
            "value": r2_mlp_full - r2_lin_full,
            "ci": _ci(boot["delta_fulldim"]),
        },
        "spearman_rho": float(rho),
        "spearman_p":   float(pval),
        "n_train": int(len(y_tr)),
        "n_test":  int(n_te),
        "n_features": int(D),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Paired bootstrap comparison of Delta-R² across paradigms (STAT-1)
# ---------------------------------------------------------------------------

def paired_delta_comparison(
    all_features: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reference_name: str,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict:
    """
    Paired bootstrap test: for each model M, test whether
        Delta-R²(reference) > Delta-R²(M)
    using the same bootstrap test-set resamples.

    Returns for each pair (reference, M):
      - mean difference, 95% CI of difference, one-tailed p-value.
    """
    # Fit all probes once; share random split
    pipes = {}
    tests = {}
    for name, (X, y) in all_features.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        D = X.shape[1]
        n_pca = min(64, D, X_tr.shape[0] - 1)

        lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        lin.fit(X_tr, y_tr)

        mlp = Pipeline([
            ("sc", StandardScaler()),
            ("pca", PCA(n_components=n_pca, random_state=random_state)),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=(128, 32), activation="relu", solver="lbfgs",
                alpha=0.01, max_iter=2000, random_state=random_state,
            )),
        ])
        mlp.fit(X_tr, y_tr)
        pipes[name] = (lin, mlp)
        tests[name] = (X_te, y_te)

    # Paired bootstrap with shared indices drawn from the test size
    # (all models should have the same n_test = 240 with identical split)
    n_te = len(next(iter(tests.values()))[1])
    rng = np.random.default_rng(random_state)

    ref_deltas = []
    comp_deltas = {n: [] for n in all_features if n != reference_name}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_te, size=n_te)
        # reference delta
        Xr, yr = tests[reference_name]
        Xrb, yrb = Xr[idx], yr[idx]
        if len(np.unique(yrb)) < 2:
            continue
        lin_r, mlp_r = pipes[reference_name]
        d_ref = float(mlp_r.score(Xrb, yrb)) - float(lin_r.score(Xrb, yrb))
        ref_deltas.append(d_ref)
        for name in comp_deltas:
            Xc, yc = tests[name]
            Xcb, ycb = Xc[idx], yc[idx]
            lin_c, mlp_c = pipes[name]
            d_c = float(mlp_c.score(Xcb, ycb)) - float(lin_c.score(Xcb, ycb))
            comp_deltas[name].append(d_c)

    ref_arr = np.array(ref_deltas)
    comparisons = {}
    for name, comp_arr in comp_deltas.items():
        comp = np.array(comp_arr)
        diff = ref_arr - comp
        mean_diff = float(np.mean(diff))
        ci = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
        # one-tailed p: fraction of resamples where ref NOT greater than comp
        p_onetail = float(np.mean(diff <= 0))
        comparisons[name] = {
            "mean_difference": mean_diff,
            "ci_95": ci,
            "p_onetail_ref_gt_comp": p_onetail,
            "significant_p05": p_onetail < 0.05,
        }

    return {
        "reference": reference_name,
        "comparisons": comparisons,
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Proper SSS for CLIP and ResNet50 (EXP-2)
# ---------------------------------------------------------------------------

def compute_contrast_sss(
    informative_features: Dict[float, np.ndarray],
    baseline_features: np.ndarray,
) -> Dict:
    """
    Compute α-sensitivity as Spearman ρ between α and the L2 distance of
    each feature vector from the mean baseline (α=0) feature vector.

    This replaces L2-norm-of-representation (which is not α-selective)
    with a contrast measure that captures α-DEPENDENT representational change.
    """
    mean_baseline = baseline_features.mean(axis=0)

    all_dists = []
    all_alphas = []
    for alpha, feats in sorted(informative_features.items()):
        dists = np.linalg.norm(feats - mean_baseline, axis=1)
        all_dists.extend(dists.tolist())
        all_alphas.extend([alpha] * len(feats))

    all_dists = np.array(all_dists)
    all_alphas = np.array(all_alphas)
    rho, pval = spearmanr(all_alphas, all_dists)

    return {
        "sss_rho_contrast": float(rho),
        "sss_p_contrast": float(pval),
        "method": "L2_distance_from_mean_baseline_feature",
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def load_or_extract_features(
    arch: str,
    data_dir: Path,
    device: str,
    cache_dir: Path,
    force_reextract: bool = False,
) -> Tuple[Dict[float, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (informative_feats_dict, informative_array, baseline_feats, labels).
      informative_feats_dict: {alpha: (N_alpha, D) array}
      informative_array:      (N_total, D)
      baseline_feats:         (N_baseline, D)
      labels:                 (N_total,)
    """
    cache_path = cache_dir / f"{arch}_features.npz"
    if cache_path.exists() and not force_reextract:
        logger.info(f"  Loading cached features: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        baseline = data["baseline"]
        # Rebuild per-alpha dict
        inf_dict = {}
        for alpha in ALPHAS:
            mask = labels == alpha
            inf_dict[alpha] = features[mask]
        return inf_dict, features, baseline, labels

    logger.info(f"  Extracting features for {arch}...")
    sys.path.insert(0, str(SCRIPT_DIR))
    from multi_architecture_analysis import (
        SDUNetAnalyzer, CLIPAnalyzer, MAEAnalyzer,
        ResNetFisherAnalyzer, AutoregressiveAnalyzer,
        resolve_informative_file,
    )
    import torch

    if arch == "sd_unet":
        analyzer = SDUNetAnalyzer(probe_timestep=500, device=device)
        feat_key = "mid_block"
    elif arch == "clip":
        analyzer = CLIPAnalyzer(device=device)
        feat_key = "cls"
    elif arch == "mae":
        analyzer = MAEAnalyzer(device=device)
        feat_key = "cls"
    elif arch == "resnet":
        analyzer = ResNetFisherAnalyzer(model_variant="resnet50", device=device)
        feat_key = "layer4"
    elif arch == "autoregressive":
        analyzer = AutoregressiveAnalyzer(device=device)
        feat_key = "hidden"
    else:
        raise ValueError(f"Unknown arch: {arch}")

    import scipy.io as sio

    def _load_and_extract(filepath):
        data = sio.loadmat(str(filepath))
        maps = data["maps"]
        if maps.ndim == 3:
            maps = np.transpose(maps, (2, 0, 1))
        maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
        stimuli = torch.tensor(maps, dtype=torch.float32)
        if arch == "resnet":
            feats = analyzer.extract_features(stimuli)
            return feats["layer4"].numpy() if isinstance(feats["layer4"], torch.Tensor) else feats["layer4"]
        elif arch == "autoregressive":
            return analyzer._extract_mean_features(stimuli)
        else:
            feats = analyzer.extract_features(stimuli)
            v = feats[feat_key]
            return v.numpy() if hasattr(v, "numpy") else v

    all_feats, all_labels = [], []
    inf_dict = {}
    for alpha in ALPHAS:
        fp = resolve_informative_file(data_dir, alpha)
        if fp is None:
            logger.warning(f"  No file for alpha={alpha}")
            continue
        f = _load_and_extract(fp)
        inf_dict[alpha] = f
        all_feats.append(f)
        all_labels.extend([alpha] * len(f))

    # Baseline (alpha=0) features for contrast SSS
    baseline_path = data_dir / "random_baseline_64x64.mat"
    if baseline_path.exists():
        baseline = _load_and_extract(baseline_path)
    else:
        logger.warning("  No baseline file found; using first alpha as proxy")
        baseline = all_feats[0]

    features = np.vstack(all_feats)
    labels = np.array(all_labels)

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, labels=labels, baseline=baseline)
    logger.info(f"  Cached to {cache_path}")

    return inf_dict, features, baseline, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Revision probing: EXP-1, EXP-2, STAT-1, INTERP-1")
    parser.add_argument("--archs", nargs="+",
                        default=["sd_unet", "clip", "mae", "resnet", "autoregressive"],
                        choices=["sd_unet", "clip", "mae", "resnet", "autoregressive"])
    parser.add_argument("--data-dir",   default="data/stimuli/")
    parser.add_argument("--output-dir", default="results/revision/")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--probe-only", action="store_true",
                        help="Use cached features; skip model loading")
    parser.add_argument("--force-reextract", action="store_true",
                        help="Ignore feature cache and re-extract")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ARCH_NAMES = {
        "sd_unet":        "Diffusion (SD UNet)",
        "clip":           "Contrastive (CLIP)",
        "mae":            "Reconstructive (MAE)",
        "resnet":         "Classification (ResNet50)",
        "autoregressive": "Autoregressive (ImageGPT)",
    }
    CONTRAST_SSS_ARCHS = {"clip", "resnet"}   # archs where SSS fix applies

    all_results = {}
    all_features_for_comparison: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for arch in args.archs:
        name = ARCH_NAMES[arch]
        logger.info(f"\n{'='*60}")
        logger.info(f"  {name}")
        logger.info(f"{'='*60}")

        try:
            inf_dict, features, baseline, labels = load_or_extract_features(
                arch=arch,
                data_dir=data_dir,
                device=device,
                cache_dir=FEATURE_CACHE_DIR,
                force_reextract=args.force_reextract,
            )
        except Exception as e:
            logger.error(f"  Feature extraction failed: {e}", exc_info=True)
            continue

        # --- Probe suite (EXP-1 + INTERP-1) ---
        logger.info(f"  Running probe suite (D={features.shape[1]}, N={len(labels)})...")
        try:
            probe_results = probe_all(features, labels, n_bootstrap=args.n_bootstrap)
        except Exception as e:
            logger.error(f"  Probing failed: {e}", exc_info=True)
            continue

        # --- Contrast SSS (EXP-2) ---
        contrast_sss = None
        if arch in CONTRAST_SSS_ARCHS:
            logger.info(f"  Computing contrast SSS (L2 distance from baseline)...")
            contrast_sss = compute_contrast_sss(inf_dict, baseline)
            logger.info(
                f"  Contrast SSS ρ = {contrast_sss['sss_rho_contrast']:.4f}"
                f"  (p = {contrast_sss['sss_p_contrast']:.2e})"
            )

        all_results[name] = {
            "arch": arch,
            "probe_results": probe_results,
            "contrast_sss": contrast_sss,
        }
        all_features_for_comparison[name] = (features, labels)

        # Print summary
        pr = probe_results
        logger.info(
            f"  linear_full    R² = {pr['linear_full']['r2']:+.4f}"
            f"  [{pr['linear_full']['ci'][0]:+.3f}, {pr['linear_full']['ci'][1]:+.3f}]"
        )
        logger.info(
            f"  linear_pca64   R² = {pr['linear_pca64']['r2']:+.4f}"
            f"  [{pr['linear_pca64']['ci'][0]:+.3f}, {pr['linear_pca64']['ci'][1]:+.3f}]"
        )
        logger.info(
            f"  mlp_pca64      R² = {pr['mlp_pca64']['r2']:+.4f}"
            f"  [{pr['mlp_pca64']['ci'][0]:+.3f}, {pr['mlp_pca64']['ci'][1]:+.3f}]"
        )
        logger.info(
            f"  mlp_full       R² = {pr['mlp_full']['r2']:+.4f}"
            f"  [{pr['mlp_full']['ci'][0]:+.3f}, {pr['mlp_full']['ci'][1]:+.3f}]"
        )
        logger.info(
            f"  Δ (original)   = {pr['delta_original']['value']:+.4f}"
            f"  [{pr['delta_original']['ci'][0]:+.3f}, {pr['delta_original']['ci'][1]:+.3f}]"
        )
        logger.info(
            f"  Δ (full-dim)   = {pr['delta_fulldim']['value']:+.4f}"
            f"  [{pr['delta_fulldim']['ci'][0]:+.3f}, {pr['delta_fulldim']['ci'][1]:+.3f}]"
        )

    # --- Paired bootstrap comparison (STAT-1) ---
    if "Diffusion (SD UNet)" in all_features_for_comparison and len(all_features_for_comparison) > 1:
        logger.info("\n=== Paired Bootstrap Comparison (STAT-1) ===")
        comparison = paired_delta_comparison(
            all_features=all_features_for_comparison,
            reference_name="Diffusion (SD UNet)",
            n_bootstrap=args.n_bootstrap,
        )
        all_results["_paired_comparison"] = comparison

        for cname, stats in comparison["comparisons"].items():
            logger.info(
                f"  SD > {cname}: mean Δ = {stats['mean_difference']:+.4f}"
                f"  95%CI [{stats['ci_95'][0]:+.3f}, {stats['ci_95'][1]:+.3f}]"
                f"  p = {stats['p_onetail_ref_gt_comp']:.4f}"
                f"  {'*' if stats['significant_p05'] else ''}"
            )

    # Save results
    out_path = output_dir / "revision_probing_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
