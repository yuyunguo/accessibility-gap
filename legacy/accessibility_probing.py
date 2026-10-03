"""
Accessibility Probing
=====================
Extracts frozen representations from each model and trains linear / MLP
regressors to decode alpha from those representations.

The gap between linear R² and MLP R² is the core metric:
  - Small gap → HOC is linearly accessible (CLIP, MAE)
  - Large gap → HOC is latently encoded but not linearly decodable (SD UNet)

Usage
-----
  # Extract features + run probing for all models:
  python scripts/accessibility_probing.py --archs sd_unet clip mae resnet autoregressive

  # GPU machine:
  python scripts/accessibility_probing.py --archs sd_unet clip mae resnet autoregressive --device cuda
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Core probing class
# ---------------------------------------------------------------------------

class LinearAccessibilityProber:
    """
    Trains a linear (Ridge) and a 2-layer MLP regressor to decode alpha
    from frozen model representations.

    Parameters
    ----------
    test_size : float
        Fraction of samples held out for evaluation (default 0.2).
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def probe(self, features: np.ndarray, labels: np.ndarray,
              n_bootstrap: int = 1000) -> Dict:
        """
        Parameters
        ----------
        features    : (N, D) array of frozen representations.
        labels      : (N,)  array of alpha values.
        n_bootstrap : number of bootstrap resamples for CIs (default 1000).

        Returns
        -------
        dict with point estimates and 95% bootstrap CIs for linear_r2, mlp_r2,
        delta_r2, spearman_rho, plus n_train, n_test, n_features.
        """
        X_tr, X_te, y_tr, y_te = train_test_split(
            features, labels,
            test_size=self.test_size,
            random_state=self.random_state,
        )

        # ---------- Linear (Ridge with z-score normalisation) ----------
        linear_pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        linear_pipe.fit(X_tr, y_tr)
        linear_r2 = float(linear_pipe.score(X_te, y_te))

        # ---------- 2-layer MLP ----------
        n_components = min(64, X_tr.shape[1], X_tr.shape[0] - 1)
        mlp_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=n_components, random_state=self.random_state)),
            ('mlp',    MLPRegressor(
                hidden_layer_sizes=(128, 32),
                activation='relu',
                solver='lbfgs',
                alpha=0.01,
                max_iter=1000,
                random_state=self.random_state,
            )),
        ])
        mlp_pipe.fit(X_tr, y_tr)
        mlp_r2 = float(mlp_pipe.score(X_te, y_te))

        # ---------- Spearman on linear predictions ----------
        y_pred_linear = linear_pipe.predict(X_te)
        rho, p = spearmanr(y_te, y_pred_linear)

        # ---------- Bootstrap CIs on test set ----------
        rng = np.random.default_rng(self.random_state)
        n_te = len(y_te)
        boot_linear, boot_mlp, boot_delta = [], [], []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_te, size=n_te)
            Xb, yb = X_te[idx], y_te[idx]
            if len(np.unique(yb)) < 2:
                continue
            bl = float(linear_pipe.score(Xb, yb))
            bm = float(mlp_pipe.score(Xb, yb))
            boot_linear.append(bl)
            boot_mlp.append(bm)
            boot_delta.append(bm - bl)

        def _ci(samples):
            a = np.array(samples)
            return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

        lin_lo, lin_hi   = _ci(boot_linear)
        mlp_lo, mlp_hi   = _ci(boot_mlp)
        dlt_lo, dlt_hi   = _ci(boot_delta)

        return {
            'linear_r2':       linear_r2,
            'linear_r2_ci':    [lin_lo, lin_hi],
            'mlp_r2':          mlp_r2,
            'mlp_r2_ci':       [mlp_lo, mlp_hi],
            'delta_r2':        mlp_r2 - linear_r2,
            'delta_r2_ci':     [dlt_lo, dlt_hi],
            'spearman_rho':    float(rho),
            'spearman_p':      float(p),
            'n_train':         int(len(y_tr)),
            'n_test':          int(n_te),
            'n_features':      int(features.shape[1]),
            'n_bootstrap':     n_bootstrap,
        }


# ---------------------------------------------------------------------------
# Feature-extraction helpers (thin wrappers around the main framework)
# ---------------------------------------------------------------------------

def extract_all_features(archs: List[str], data_dir: str,
                          max_stimuli: Optional[int],
                          device: str) -> Dict[str, Dict]:
    """
    Instantiate each requested analyzer, run analyze_psychometric_set,
    and return a dict keyed by model name containing 'features' and 'labels'.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from multi_architecture_analysis import (
        SDUNetAnalyzer, CLIPAnalyzer, MAEAnalyzer,
        ResNetFisherAnalyzer, AutoregressiveAnalyzer,
        resolve_informative_file,
    )
    import torch
    import scipy.io as sio

    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    registry = {}
    if "sd_unet" in archs:
        registry["Diffusion (SD UNet)"] = lambda: SDUNetAnalyzer(
            probe_timestep=500, device=device)
    if "clip" in archs:
        registry["Contrastive (CLIP)"] = lambda: CLIPAnalyzer(device=device)
    if "mae" in archs:
        registry["Reconstructive (MAE)"] = lambda: MAEAnalyzer(device=device)
    if "resnet" in archs:
        registry["Classification (ResNet50)"] = lambda: ResNetFisherAnalyzer(
            model_variant="resnet50", device=device)
    if "autoregressive" in archs:
        registry["Autoregressive (ImageGPT)"] = lambda: AutoregressiveAnalyzer(
            device=device)

    results = {}
    for name, factory in registry.items():
        logger.info(f"\n=== {name} ===")
        try:
            analyzer = factory()
            res = analyzer.analyze_psychometric_set(
                data_dir=data_dir, alphas=alphas, max_stimuli=max_stimuli)
            if 'features' not in res:
                logger.warning(f"  {name}: no 'features' key — skipping probing")
                continue
            # Consolidate: pick first available feature key
            feat_dict = res['features']
            feat_key = next(k for k in feat_dict if k != 'labels')
            features = feat_dict[feat_key]     # (N, D)
            labels   = feat_dict['labels']     # (N,)
            results[name] = {
                'features': features,
                'labels':   labels,
                'raw_results': {k: v for k, v in res.items() if k != 'features'},
            }
            logger.info(f"  features: {features.shape}, labels: {labels.shape}")
        except Exception as e:
            logger.error(f"  Failed: {e}", exc_info=True)

    return results


# ---------------------------------------------------------------------------
# ResNet: extract avgpool features separately (no 'features' key by default)
# ---------------------------------------------------------------------------

def _extract_resnet_features(data_dir: str, max_stimuli: Optional[int],
                               device: str) -> Tuple[np.ndarray, np.ndarray]:
    import sys, torch
    sys.path.insert(0, str(Path(__file__).parent))
    from multi_architecture_analysis import ResNetFisherAnalyzer, resolve_informative_file

    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    analyzer = ResNetFisherAnalyzer(model_variant="resnet50", device=device)
    data_dir = Path(data_dir)

    all_feats, all_labels = [], []
    for alpha in alphas:
        fp = resolve_informative_file(data_dir, alpha)
        if fp is None:
            continue
        stimuli = analyzer.load_stimuli(str(fp), max_stimuli=max_stimuli)
        feats = analyzer.extract_features(stimuli)
        # Use the deepest layer (layer4) as the representation
        all_feats.append(feats['layer4'].numpy())
        all_labels.extend([alpha] * len(stimuli))

    return np.vstack(all_feats), np.array(all_labels)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Accessibility probing for all models")
    parser.add_argument("--archs", nargs="+",
                        default=["sd_unet", "clip", "mae", "resnet", "autoregressive"],
                        help="Models to probe")
    parser.add_argument("--data-dir", default="data/stimuli/")
    parser.add_argument("--output-dir", default="results/probing/")
    parser.add_argument("--max-stimuli", type=int, default=None)
    parser.add_argument("--device", default=None,
                        help="Torch device (default: auto-detect cuda/cpu)")
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probing_archs = [a for a in args.archs if a != "resnet"]
    feature_data = extract_all_features(
        probing_archs, args.data_dir, args.max_stimuli, device)

    # ResNet needs special handling (features not stored in analyze_psychometric_set)
    if "resnet" in args.archs:
        logger.info("\n=== Classification (ResNet50) ===")
        try:
            feats, labels = _extract_resnet_features(
                args.data_dir, args.max_stimuli, device)
            feature_data["Classification (ResNet50)"] = {
                'features': feats, 'labels': labels, 'raw_results': {}}
            logger.info(f"  features: {feats.shape}")
        except Exception as e:
            logger.error(f"  ResNet feature extraction failed: {e}", exc_info=True)

    prober = LinearAccessibilityProber(test_size=0.2, random_state=42)
    probing_results = {}

    logger.info("\n=== Probing Results ===")
    header = f"{'Model':<30} {'Linear R²':>10} {'MLP R²':>10} {'Gap':>10} {'Spearman ρ':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for name, data in feature_data.items():
        try:
            res = prober.probe(data['features'], data['labels'])
            probing_results[name] = res
            gap = res['delta_r2']
            lin_ci = res['linear_r2_ci']
            dlt_ci = res['delta_r2_ci']
            logger.info(
                f"{name:<30} {res['linear_r2']:>7.3f} [{lin_ci[0]:+.3f},{lin_ci[1]:+.3f}]"
                f"  {res['mlp_r2']:>7.3f}  Δ={gap:+.3f} [{dlt_ci[0]:+.3f},{dlt_ci[1]:+.3f}]"
            )
        except Exception as e:
            logger.error(f"  Probing failed for {name}: {e}", exc_info=True)

    # Save — merge with any existing results so partial runs accumulate
    out_path = output_dir / "accessibility_probing_results.json"
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(probing_results)
        probing_results = existing
    with open(out_path, "w") as f:
        json.dump(probing_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
