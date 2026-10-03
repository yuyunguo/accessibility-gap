"""
Same-Backbone Controlled Study (EXP-A)
=======================================
Addresses the causal confound identified in peer review: existing comparisons
vary architecture, training data, AND objective simultaneously.

This script holds architecture and training data constant (ViT-B, ImageNet-1K)
and varies only the training objective:

  Model                   Objective          HuggingFace ID
  ─────────────────────── ────────────────── ──────────────────────────────
  MAE-ViT-B/16            Reconstructive     facebook/vit-mae-base
  DINO-ViT-B/16           Self-supervised    facebook/dino-vitb16
  Supervised ViT-B/16     Classification     google/vit-base-patch16-224

CLIP-ViT-B/32 is included as cross-architecture comparison (different patch
size, same base architecture family).

Usage
-----
  python scripts/same_backbone_probing.py --device cuda
  python scripts/same_backbone_probing.py --probe-only      # uses cached features
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR   = Path(__file__).parent
RESULTS_DIR  = SCRIPT_DIR.parent / "results" / "revision"
CACHE_DIR    = RESULTS_DIR / "feature_cache"
DATA_DIR_DEFAULT = SCRIPT_DIR.parent / "data" / "stimuli"

ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# PIL helper (shared by all ViT-family extractors)
# ---------------------------------------------------------------------------

def _to_pil_list(images: torch.Tensor) -> List:
    """Convert (N, H, W) or (N, C, H, W) float tensor [0,1] to list of PIL Images."""
    from PIL import Image
    out = []
    for img in images:
        arr = (img.clamp(0, 1) * 255).numpy().astype(np.uint8)
        if arr.ndim == 2:                        # (H, W) grayscale
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[0] in (1, 3):             # (C, H, W)
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        out.append(Image.fromarray(arr))
    return out


# ---------------------------------------------------------------------------
# Extractor: DINO-ViT-B/16
# ---------------------------------------------------------------------------

class DINOExtractor:
    """
    Facebook DINO self-supervised ViT-B/16 trained on ImageNet-1K.
    Extracts the CLS token from the final transformer layer.
    Output: (N, 768)
    """

    MODEL_ID = "facebook/dino-vitb16"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModel
        logger.info(f"Loading DINO: {self.MODEL_ID}")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).to(device).eval()
        logger.info("  DINO ViT-B/16 ready")

    def extract(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        all_cls = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="DINO features"):
                pils = _to_pil_list(images[i:i + batch_size])
                inputs = self.processor(images=pils, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self.model(**inputs)
                cls = out.last_hidden_state[:, 0, :]   # CLS token
                all_cls.append(cls.float().cpu().numpy())
        return np.vstack(all_cls)


# ---------------------------------------------------------------------------
# Extractor: Supervised ViT-B/16
# ---------------------------------------------------------------------------

class SupervisedViTExtractor:
    """
    Google supervised ViT-B/16 trained on ImageNet-1K with classification CE loss.
    Extracts the CLS token (penultimate layer output before classification head).
    Output: (N, 768)
    """

    MODEL_ID = "google/vit-base-patch16-224"

    def __init__(self, device: str = "cpu"):
        from transformers import AutoImageProcessor, ViTModel
        logger.info(f"Loading Supervised ViT: {self.MODEL_ID}")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        # Load base model (encoder only, no classification head) for clean features
        self.model = ViTModel.from_pretrained(self.MODEL_ID).to(device).eval()
        logger.info("  Supervised ViT-B/16 ready")

    def extract(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        all_cls = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="SupViT features"):
                pils = _to_pil_list(images[i:i + batch_size])
                inputs = self.processor(images=pils, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                out = self.model(**inputs)
                cls = out.last_hidden_state[:, 0, :]   # CLS token
                all_cls.append(cls.float().cpu().numpy())
        return np.vstack(all_cls)


# ---------------------------------------------------------------------------
# Stimulus loading
# ---------------------------------------------------------------------------

def resolve_informative_file(data_dir: Path, alpha: float) -> Optional[Path]:
    """Find the .mat file for a given alpha level (mirrors multi_architecture_analysis)."""
    compact = str(alpha).replace('.', '')
    padded  = f"{int(round(alpha * 100)):03d}"
    legacy  = f"{int(round(alpha * 10)):02d}"
    candidates = [
        data_dir / f"informative_alpha_{compact}_64x64.mat",
        data_dir / f"informative_alpha_{padded}_64x64.mat",
        data_dir / f"informative_alpha_{legacy}.mat",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_stimuli(filepath: Path) -> torch.Tensor:
    """Load .mat stimuli → (N, H, W) float32 in [0,1]."""
    data = sio.loadmat(str(filepath))
    maps = data["maps"]
    if maps.ndim == 3:
        maps = np.transpose(maps, (2, 0, 1))
    maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
    return torch.tensor(maps.astype(np.float32))


def extract_all_features(
    extractor,
    data_dir: Path,
    cache_path: Path,
    force: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features for all alpha levels.
    Returns: (features, labels, baseline_features)
    """
    if cache_path.exists() and not force:
        logger.info(f"  Loading cached features: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["features"], data["labels"], data["baseline"]

    all_feats, all_labels = [], []
    for alpha in ALPHAS:
        fp = resolve_informative_file(data_dir, alpha)
        if fp is None:
            logger.warning(f"  No stimulus file for alpha={alpha}")
            continue
        stimuli = load_stimuli(fp)
        feats = extractor.extract(stimuli)
        all_feats.append(feats)
        all_labels.extend([alpha] * len(feats))
        logger.info(f"    alpha={alpha}: {len(feats)} stimuli, feature dim={feats.shape[1]}")

    baseline_path = data_dir / "random_baseline_64x64.mat"
    if baseline_path.exists():
        baseline_stimuli = load_stimuli(baseline_path)
        baseline = extractor.extract(baseline_stimuli)
    else:
        logger.warning("  No baseline file; using first alpha block as proxy")
        baseline = all_feats[0]

    features = np.vstack(all_feats)
    labels   = np.array(all_labels)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, features=features, labels=labels, baseline=baseline)
    logger.info(f"  Cached to {cache_path}")
    return features, labels, baseline


# ---------------------------------------------------------------------------
# Probe suite (mirrors revision_probing.probe_all exactly)
# ---------------------------------------------------------------------------

def probe_all(
    features: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Dict:
    X_tr, X_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    n_te = len(y_te)
    D    = features.shape[1]
    n_pca = min(64, D, X_tr.shape[0] - 1)

    pipe_lin_full = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe_lin_full.fit(X_tr, y_tr)

    pipe_lin_pca = Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe_lin_pca.fit(X_tr, y_tr)

    pipe_mlp_pca = Pipeline([
        ("sc", StandardScaler()),
        ("pca", PCA(n_components=n_pca, random_state=random_state)),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 32), activation="relu", solver="lbfgs",
            alpha=0.01, max_iter=2000, random_state=random_state,
        )),
    ])
    pipe_mlp_pca.fit(X_tr, y_tr)

    pipe_mlp_full = Pipeline([
        ("sc", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 32), activation="relu", solver="lbfgs",
            alpha=0.1, max_iter=2000, random_state=random_state,
        )),
    ])
    pipe_mlp_full.fit(X_tr, y_tr)

    rng  = np.random.default_rng(random_state)
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
        boot["delta_original"].append(b_mp - b_lf)
        boot["delta_fulldim"].append(b_mf - b_lf)

    def _ci(samples):
        a = np.array(samples)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    rho, pval = spearmanr(y_te, pipe_lin_full.predict(X_te))
    r2_lf = float(pipe_lin_full.score(X_te, y_te))
    r2_lp = float(pipe_lin_pca.score(X_te, y_te))
    r2_mp = float(pipe_mlp_pca.score(X_te, y_te))
    r2_mf = float(pipe_mlp_full.score(X_te, y_te))

    return {
        "linear_full":    {"r2": r2_lf, "ci": _ci(boot["lin_full"])},
        "linear_pca64":   {"r2": r2_lp, "ci": _ci(boot["lin_pca"])},
        "mlp_pca64":      {"r2": r2_mp, "ci": _ci(boot["mlp_pca"])},
        "mlp_full":       {"r2": r2_mf, "ci": _ci(boot["mlp_full"])},
        "delta_original": {"value": r2_mp - r2_lf, "ci": _ci(boot["delta_original"])},
        "delta_fulldim":  {"value": r2_mf - r2_lf, "ci": _ci(boot["delta_fulldim"])},
        "spearman_rho": float(rho),
        "spearman_p":   float(pval),
        "n_train": int(len(y_tr)),
        "n_test":  int(n_te),
        "n_features": int(D),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODELS = {
    "dino":         ("DINO (ViT-B/16)",          DINOExtractor,          "dino_features.npz"),
    "supervised_vit": ("Supervised ViT-B/16",     SupervisedViTExtractor, "supervised_vit_features.npz"),
}

# Existing cached results to include in the same-backbone table
EXISTING_CACHE = {
    "mae":  ("MAE (ViT-B/16)",    "mae_features.npz"),
    "clip": ("CLIP (ViT-B/32)",   "clip_features.npz"),
}


def print_result(name: str, pr: Dict):
    logger.info(f"\n  [{name}]")
    logger.info(f"  linear_full  R² = {pr['linear_full']['r2']:+.4f}  "
                f"[{pr['linear_full']['ci'][0]:+.3f}, {pr['linear_full']['ci'][1]:+.3f}]")
    logger.info(f"  mlp_full     R² = {pr['mlp_full']['r2']:+.4f}  "
                f"[{pr['mlp_full']['ci'][0]:+.3f}, {pr['mlp_full']['ci'][1]:+.3f}]")
    logger.info(f"  Δ_full       = {pr['delta_fulldim']['value']:+.4f}  "
                f"[{pr['delta_fulldim']['ci'][0]:+.3f}, {pr['delta_fulldim']['ci'][1]:+.3f}]")


def main():
    parser = argparse.ArgumentParser(description="Same-backbone controlled study (EXP-A)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--data-dir",   default=str(DATA_DIR_DEFAULT))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    data_dir = Path(args.data_dir)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # ---- New models (DINO, Supervised ViT) ----
    for key in args.models:
        display_name, ExtractorClass, cache_fname = MODELS[key]
        logger.info(f"\n{'='*60}")
        logger.info(f"  {display_name}")
        logger.info(f"{'='*60}")

        cache_path = CACHE_DIR / cache_fname

        if args.probe_only and not cache_path.exists():
            logger.warning(f"  Cache not found for {key}; skipping (run without --probe-only first)")
            continue

        try:
            extractor = ExtractorClass(device=device) if not args.probe_only else None
            features, labels, baseline = extract_all_features(
                extractor, data_dir, cache_path, force=args.force_reextract
            )
        except Exception as e:
            logger.error(f"  Extraction failed: {e}", exc_info=True)
            continue

        logger.info(f"  Running probe suite (D={features.shape[1]}, N={len(labels)})...")
        try:
            pr = probe_all(features, labels, n_bootstrap=args.n_bootstrap)
        except Exception as e:
            logger.error(f"  Probing failed: {e}", exc_info=True)
            continue

        all_results[display_name] = {"arch": key, "probe_results": pr}
        print_result(display_name, pr)

    # ---- Load existing cached results for comparison ----
    logger.info(f"\n{'='*60}")
    logger.info("  Loading existing ViT-B results for comparison")
    logger.info(f"{'='*60}")
    for key, (display_name, cache_fname) in EXISTING_CACHE.items():
        cache_path = CACHE_DIR / cache_fname
        if not cache_path.exists():
            logger.warning(f"  Cached features not found for {key} ({cache_path}); skipping")
            continue
        data     = np.load(cache_path, allow_pickle=True)
        features = data["features"]
        labels   = data["labels"]
        logger.info(f"  Running probes for {display_name} (D={features.shape[1]}, N={len(labels)})...")
        try:
            pr = probe_all(features, labels, n_bootstrap=args.n_bootstrap)
        except Exception as e:
            logger.error(f"  Probing failed for {display_name}: {e}", exc_info=True)
            continue
        all_results[display_name] = {"arch": key, "probe_results": pr}
        print_result(display_name, pr)

    # ---- Save ----
    out_path = RESULTS_DIR / "same_backbone_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved to {out_path}")

    # ---- Summary table ----
    logger.info("\n" + "="*60)
    logger.info("  SAME-BACKBONE SUMMARY")
    logger.info("="*60)
    logger.info(f"  {'Model':<25} {'Lin R²':>8}  {'MLP R²':>8}  {'ΔR²_full':>10}  CI")
    logger.info("  " + "-"*65)
    row_order = [
        "MAE (ViT-B/16)", "DINO (ViT-B/16)", "Supervised ViT-B/16", "CLIP (ViT-B/32)"
    ]
    for name in row_order + [k for k in all_results if k not in row_order]:
        if name not in all_results:
            continue
        pr = all_results[name]["probe_results"]
        lf = pr["linear_full"]["r2"]
        mf = pr["mlp_full"]["r2"]
        d  = pr["delta_fulldim"]["value"]
        ci = pr["delta_fulldim"]["ci"]
        logger.info(f"  {name:<25} {lf:>+8.4f}  {mf:>+8.4f}  {d:>+10.4f}  "
                    f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")


if __name__ == "__main__":
    main()
