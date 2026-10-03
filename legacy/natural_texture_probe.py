"""
Natural Texture Probe (EXP-C)
================================
Addresses reviewer concern: external validity limited to synthetic binary textures.

Tests whether the qualitative ordering of accessibility gaps (SD UNet >> MAE/CLIP/DINO)
holds when the target statistic is computed from natural image textures rather than
synthetic BTC stimuli.

Dataset: DTD (Describable Textures Dataset) — 5640 images, 47 categories.
         64x64 grayscale center crops match the BTC stimulus spatial scale.

Target: 4-point correlation coefficient (gamma) at the 2x2 square configuration
        on binarized patches — direct natural-image analog of the BTC alpha parameter.
        gamma = E[b[i,j] * b[i+1,j] * b[i,j+1] * b[i+1,j+1]] where b = sign(pixel - median)

Success criterion: qualitative ordering preserved (SD UNet gap > MAE/CLIP gaps)
                   even if absolute magnitudes differ from BTC results.

Models: clip, mae, sd_unet, dino, supervised_vit, resnet
Output: results/revision/natural_texture_results.json

Usage
-----
  python scripts/natural_texture_probe.py --device cuda
  python scripts/natural_texture_probe.py --probe-only   # reuse cached features
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

SCRIPT_DIR  = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "revision"
CACHE_DIR   = RESULTS_DIR / "feature_cache" / "natural"
DATA_DIR    = SCRIPT_DIR.parent / "data"

PATCH_SIZE = 64   # match BTC stimulus spatial scale


# ---------------------------------------------------------------------------
# 4-point correlation — gamma (analog of alpha in BTC stimuli)
# ---------------------------------------------------------------------------

def compute_gamma(patch: np.ndarray) -> float:
    """
    4-point correlation at the 2x2 square configuration on a binarized patch.

    Binarize at median (pixels >= median -> +1, else -1), then compute
        gamma = E[b[i,j] * b[i+1,j] * b[i,j+1] * b[i+1,j+1]]
    over all valid pixel quadruples.  Returns gamma in [-1, 1]; independent
    (memoryless) textures give gamma ~ 0.
    """
    b = np.where(patch >= np.median(patch), 1.0, -1.0)
    return float(np.mean(b[:-1, :-1] * b[1:, :-1] * b[:-1, 1:] * b[1:, 1:]))


# ---------------------------------------------------------------------------
# DTD loading
# ---------------------------------------------------------------------------

def load_dtd(data_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all DTD splits (train + val + test), center-crop to PATCH_SIZE grayscale.

    Returns
    -------
    patches : (N, PATCH_SIZE, PATCH_SIZE) float32 in [0, 1]
    gammas  : (N,) float32 — 4-point correlation per patch
    """
    from torchvision.datasets import DTD
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.CenterCrop(PATCH_SIZE),
        transforms.ToTensor(),   # -> (1, H, W) in [0, 1]
    ])

    patches_list, gammas_list = [], []
    for split in ("train", "val", "test"):
        ds = DTD(root=str(data_root), split=split, transform=transform, download=True)
        for img_t, _ in tqdm(ds, desc=f"DTD {split}", leave=False):
            patch = img_t.squeeze(0).numpy()   # (H, W) float32
            patches_list.append(patch)
            gammas_list.append(compute_gamma(patch))

    patches = np.stack(patches_list).astype(np.float32)
    gammas  = np.array(gammas_list, dtype=np.float32)
    logger.info(
        f"DTD: {len(patches)} patches  "
        f"gamma mean={gammas.mean():.4f}  std={gammas.std():.4f}  "
        f"range=[{gammas.min():.3f}, {gammas.max():.3f}]"
    )
    return patches, gammas


# ---------------------------------------------------------------------------
# PIL conversion helper (shared by ViT-family extractors)
# ---------------------------------------------------------------------------

def _to_pil(images: torch.Tensor) -> List:
    """(N, H, W) or (N, C, H, W) float tensor [0,1] -> list of RGB PIL Images."""
    from PIL import Image
    out = []
    for img in images:
        arr = (img.clamp(0, 1) * 255).numpy().astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
        out.append(Image.fromarray(arr))
    return out


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class CLIPExtractor:
    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(self, device):
        from transformers import CLIPModel, CLIPProcessor
        logger.info(f"Loading CLIP ({self.MODEL_ID})")
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_ID)
        self.model = CLIPModel.from_pretrained(self.MODEL_ID).to(device).eval()

    def extract(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="CLIP", leave=False):
                inp = self.processor(images=_to_pil(images[i:i+batch_size]),
                                     return_tensors="pt")
                pv = inp["pixel_values"].to(self.device)
                vision_out = self.model.vision_model(pixel_values=pv)
                feats = self.model.visual_projection(vision_out.pooler_output)
                out.append(feats.float().cpu().numpy())
        return np.vstack(out)


class MAEExtractor:
    MODEL_ID = "facebook/vit-mae-base"

    def __init__(self, device):
        from transformers import AutoImageProcessor, ViTMAEModel
        logger.info(f"Loading MAE ({self.MODEL_ID})")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = ViTMAEModel.from_pretrained(self.MODEL_ID).to(device).eval()

    def extract(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="MAE", leave=False):
                inp = self.processor(images=_to_pil(images[i:i+batch_size]),
                                     return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                cls = self.model(**inp).last_hidden_state[:, 0, :]
                out.append(cls.float().cpu().numpy())
        return np.vstack(out)


class DINOExtractor:
    MODEL_ID = "facebook/dino-vitb16"

    def __init__(self, device):
        from transformers import AutoImageProcessor, AutoModel
        logger.info(f"Loading DINO ({self.MODEL_ID})")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).to(device).eval()

    def extract(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="DINO", leave=False):
                inp = self.processor(images=_to_pil(images[i:i+batch_size]),
                                     return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                cls = self.model(**inp).last_hidden_state[:, 0, :]
                out.append(cls.float().cpu().numpy())
        return np.vstack(out)


class SupervisedViTExtractor:
    MODEL_ID = "google/vit-base-patch16-224"

    def __init__(self, device):
        from transformers import AutoImageProcessor, ViTModel
        logger.info(f"Loading Supervised ViT ({self.MODEL_ID})")
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = ViTModel.from_pretrained(self.MODEL_ID).to(device).eval()

    def extract(self, images: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="SupViT", leave=False):
                inp = self.processor(images=_to_pil(images[i:i+batch_size]),
                                     return_tensors="pt")
                inp = {k: v.to(self.device) for k, v in inp.items()}
                cls = self.model(**inp).last_hidden_state[:, 0, :]
                out.append(cls.float().cpu().numpy())
        return np.vstack(out)


class ResNetExtractor:
    def __init__(self, device):
        import torchvision.models as models
        import torchvision.transforms as T
        logger.info("Loading ResNet50 (ImageNet-1K)")
        self.device = device
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(m.children())[:-1]).to(device).eval()

    def extract(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        import torch.nn.functional as F
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="ResNet50", leave=False):
                batch = images[i:i+batch_size]            # (B, H, W)
                rgb = batch.unsqueeze(1).repeat(1, 3, 1, 1).float()  # (B, 3, H, W)
                rgb = F.interpolate(rgb, size=(224, 224), mode="bilinear",
                                    align_corners=False)
                rgb = torch.stack([self.norm(x) for x in rgb]).to(self.device)
                feats = self.model(rgb).squeeze(-1).squeeze(-1)
                out.append(feats.float().cpu().numpy())
        return np.vstack(out)


class SDUNetExtractor:
    MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(self, device, probe_timestep: int = 500):
        import torch.nn.functional as F
        from diffusers import StableDiffusionPipeline
        logger.info(f"Loading SD UNet ({self.MODEL_ID}, t={probe_timestep})")
        self.device = device
        self.probe_timestep = probe_timestep
        self._F = F
        self._feat = None
        dtype = torch.float16 if "cuda" in str(device) else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=dtype,
            safety_checker=None, requires_safety_checker=False,
        )
        self.vae = pipe.vae.to(device).eval()
        self.unet = pipe.unet.to(device).eval()
        self.scheduler = pipe.scheduler
        self.scheduler.set_timesteps(1000)
        self._dtype = dtype
        null_ids = pipe.tokenizer(
            [""], padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        self._null_emb = pipe.text_encoder(null_ids)[0].detach().to(device)
        self.unet.mid_block.register_forward_hook(
            lambda m, inp, out: setattr(
                self, "_feat", out[0] if isinstance(out, (tuple, list)) else out
            )
        )

    def _prep(self, img: torch.Tensor) -> torch.Tensor:
        F = self._F
        if img.ndim == 2:
            img = img.unsqueeze(0).repeat(3, 1, 1)
        elif img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if img.shape[-1] != 512:
            img = F.interpolate(img.unsqueeze(0), size=(512, 512),
                                mode="bilinear", align_corners=False).squeeze(0)
        return (2.0 * img - 1.0).to(self._dtype)

    def extract(self, images: torch.Tensor, batch_size: int = 4) -> np.ndarray:
        import torch.nn.functional as F
        out = []
        with torch.no_grad():
            for i in tqdm(range(0, len(images), batch_size), desc="SD UNet", leave=False):
                batch = images[i:i+batch_size]
                x = torch.stack([self._prep(img) for img in batch]).to(self.device)
                latents = (self.vae.encode(x).latent_dist.mean
                           * self.vae.config.scaling_factor)
                noise = torch.randn_like(latents)
                t = torch.full((len(batch),), self.probe_timestep,
                               device=self.device, dtype=torch.long)
                noisy = self.scheduler.add_noise(latents, noise, t)
                null_c = self._null_emb.expand(len(batch), -1, -1).to(self._dtype)
                self.unet(noisy, t, encoder_hidden_states=null_c)
                feat = F.adaptive_avg_pool2d(self._feat, 1).squeeze(-1).squeeze(-1)
                out.append(feat.float().cpu())
        return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Probe suite (linear-full + MLP-full, matching main paper ΔR²_full metric)
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

    pipe_lin = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    pipe_lin.fit(X_tr, y_tr)

    pipe_mlp = Pipeline([
        ("sc",  StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 32), activation="relu", solver="lbfgs",
            alpha=0.1, max_iter=2000, random_state=random_state,
        )),
    ])
    pipe_mlp.fit(X_tr, y_tr)

    rng = np.random.default_rng(random_state)
    boot_lin, boot_mlp, boot_delta = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_te, size=n_te)
        Xb, yb = X_te[idx], y_te[idx]
        bl = float(pipe_lin.score(Xb, yb))
        bm = float(pipe_mlp.score(Xb, yb))
        boot_lin.append(bl)
        boot_mlp.append(bm)
        boot_delta.append(bm - bl)

    def _ci(s):
        a = np.array(s)
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    r2_lin = float(pipe_lin.score(X_te, y_te))
    r2_mlp = float(pipe_mlp.score(X_te, y_te))
    rho, pval = spearmanr(y_te, pipe_lin.predict(X_te))

    return {
        "linear_full":   {"r2": r2_lin, "ci": _ci(boot_lin)},
        "mlp_full":      {"r2": r2_mlp, "ci": _ci(boot_mlp)},
        "delta_fulldim": {"value": r2_mlp - r2_lin, "ci": _ci(boot_delta)},
        "spearman_rho":  float(rho),
        "spearman_p":    float(pval),
        "n_train": int(len(y_tr)),
        "n_test":  int(n_te),
        "n_features": int(features.shape[1]),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Models registry
# ---------------------------------------------------------------------------

MODELS = {
    "clip":           ("Contrastive (CLIP)",       CLIPExtractor,          "nat_clip.npz"),
    "mae":            ("Reconstructive (MAE)",      MAEExtractor,           "nat_mae.npz"),
    "dino":           ("DINO (ViT-B/16)",           DINOExtractor,          "nat_dino.npz"),
    "supervised_vit": ("Supervised ViT-B/16",       SupervisedViTExtractor, "nat_supvit.npz"),
    "resnet":         ("Classification (ResNet50)", ResNetExtractor,        "nat_resnet.npz"),
    "sd_unet":        ("Diffusion (SD UNet)",       SDUNetExtractor,        "nat_sd_unet.npz"),
}

DISPLAY_ORDER = [
    "Diffusion (SD UNet)", "Reconstructive (MAE)", "Contrastive (CLIP)",
    "DINO (ViT-B/16)", "Supervised ViT-B/16", "Classification (ResNet50)",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Natural texture probe (EXP-C)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--probe-only", action="store_true",
                        help="Skip feature extraction; use cached .npz files")
    parser.add_argument("--force-reextract", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load / cache DTD patches and gamma targets ----
    dtd_cache = RESULTS_DIR / "dtd_patches_gamma.npz"
    if dtd_cache.exists():
        logger.info(f"Loading cached DTD data: {dtd_cache}")
        d = np.load(dtd_cache)
        patches, gammas = d["patches"], d["gammas"]
        logger.info(f"  {len(patches)} patches  gamma std={gammas.std():.4f}")
    else:
        patches, gammas = load_dtd(DATA_DIR)
        np.savez_compressed(dtd_cache, patches=patches, gammas=gammas)
        logger.info(f"Saved DTD cache: {dtd_cache}")

    if gammas.std() < 0.005:
        logger.warning(
            f"gamma std={gammas.std():.5f} is very low — regression may be uninformative. "
            "Consider using a larger patch size or multi-offset gamma."
        )

    patches_t = torch.tensor(patches)   # (N, H, W) float32 [0, 1]

    results: Dict = {
        "_meta": {
            "dataset": "DTD (train+val+test)",
            "n_images": int(len(patches)),
            "patch_size": PATCH_SIZE,
            "target": "4-point correlation gamma, 2x2 square config, binarized at median",
            "gamma_mean":  float(gammas.mean()),
            "gamma_std":   float(gammas.std()),
            "gamma_min":   float(gammas.min()),
            "gamma_max":   float(gammas.max()),
        }
    }

    for key in args.models:
        display_name, ExtractorClass, cache_fname = MODELS[key]
        cache_path = CACHE_DIR / cache_fname

        logger.info(f"\n{'='*60}")
        logger.info(f"  {display_name}")
        logger.info(f"{'='*60}")

        if cache_path.exists() and not args.force_reextract:
            logger.info(f"  Loading cached features: {cache_path}")
            features = np.load(cache_path)["features"]
        elif args.probe_only:
            logger.warning(f"  No cache for {key}; skipping (run without --probe-only first)")
            continue
        else:
            extractor = ExtractorClass(device=device)
            features = extractor.extract(patches_t)
            np.savez_compressed(cache_path, features=features)
            logger.info(f"  Saved features {features.shape} -> {cache_path}")
            del extractor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"  Probing (D={features.shape[1]}, N={len(features)})...")
        pr = probe_all(features, gammas, n_bootstrap=args.n_bootstrap)
        results[display_name] = {"arch": key, "probe_results": pr}

        lf = pr["linear_full"]["r2"]
        mf = pr["mlp_full"]["r2"]
        dv = pr["delta_fulldim"]["value"]
        ci = pr["delta_fulldim"]["ci"]
        logger.info(
            f"  linear R²={lf:+.4f} [{pr['linear_full']['ci'][0]:+.3f},"
            f"{pr['linear_full']['ci'][1]:+.3f}]  "
            f"MLP R²={mf:+.4f}  ΔR²={dv:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
        )

    out_path = RESULTS_DIR / "natural_texture_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved -> {out_path}")

    # ---- Summary table ----
    logger.info("\n" + "="*70)
    logger.info("  EXP-C: Natural Texture Probe (DTD, gamma target)")
    logger.info("="*70)
    logger.info(f"  {'Model':<30} {'Lin R²':>8}  {'MLP R²':>8}  {'ΔR²_full':>10}  95% CI")
    logger.info("  " + "-"*70)
    for name in DISPLAY_ORDER + [k for k in results if k not in DISPLAY_ORDER and not k.startswith("_")]:
        if name not in results:
            continue
        pr = results[name]["probe_results"]
        lf = pr["linear_full"]["r2"]
        mf = pr["mlp_full"]["r2"]
        dv = pr["delta_fulldim"]["value"]
        ci = pr["delta_fulldim"]["ci"]
        logger.info(
            f"  {name:<30} {lf:>+8.4f}  {mf:>+8.4f}  {dv:>+10.4f}  "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        )


if __name__ == "__main__":
    main()
