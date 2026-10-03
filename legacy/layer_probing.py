"""
Layer-wise Probing for CLIP and MAE
=====================================
Registers forward hooks on every transformer encoder block (layers 0–11),
captures the CLS token after each block, and runs LinearAccessibilityProber
at each depth.

Outputs
-------
  results/probing/layer_probing_results.json
  results/probing/layer_probing_figure.{pdf,png}

Usage
-----
  python scripts/layer_probing.py --archs clip mae --device cuda
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Insert scripts/ on path so we can import siblings ────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from accessibility_probing import LinearAccessibilityProber
from multi_architecture_analysis import resolve_informative_file


# ── Stimulus loading (mirrors what analyzers do) ──────────────────────────────

def load_stimuli_mat(path: str, max_stimuli: Optional[int] = None) -> torch.Tensor:
    data = sio.loadmat(path)
    key = [k for k in data if not k.startswith("_")][0]
    arr = data[key].astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    if arr.ndim == 3:           # (H, W, N) → (N, H, W)
        arr = arr.transpose(2, 0, 1)
    elif arr.ndim == 4:         # (N, H, W, C) or similar
        pass
    if max_stimuli is not None:
        arr = arr[:max_stimuli]
    return torch.from_numpy(arr)


def load_all_stimuli(data_dir: str, alphas: List[float],
                     max_stimuli: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (all_images_as_float32_CHW, labels)."""
    data_dir = Path(data_dir)
    all_imgs, all_labels = [], []
    for alpha in alphas:
        fp = resolve_informative_file(data_dir, alpha)
        if fp is None:
            logger.warning(f"  Missing stimuli for alpha={alpha}")
            continue
        imgs = load_stimuli_mat(str(fp), max_stimuli=max_stimuli)   # (N, H, W)
        all_imgs.append(imgs)
        all_labels.extend([alpha] * len(imgs))
    return torch.cat(all_imgs, dim=0), np.array(all_labels, dtype=np.float32)


def _to_pil_list(images: torch.Tensor):
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


# ── Layer-wise feature extraction ────────────────────────────────────────────

def extract_clip_layer_features(images: torch.Tensor, device: str,
                                  batch_size: int = 32) -> Dict[int, np.ndarray]:
    """
    Returns {layer_idx: (N, 768)} CLS-token array for each of the 12
    CLIP ViT-B/32 encoder blocks.
    """
    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    n_layers = len(model.vision_model.encoder.layers)
    logger.info(f"  CLIP: {n_layers} encoder blocks")

    # Accumulate per-layer CLS tokens across batches
    layer_buffers: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}

    hooks = []
    for i, block in enumerate(model.vision_model.encoder.layers):
        def _make_hook(idx):
            def _hook(module, input, output):
                # Newer transformers may return a tensor directly or a tuple
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                cls = hidden[:, 0, :].float().cpu()
                layer_buffers[idx].append(cls)
            return _hook
        hooks.append(block.register_forward_hook(_make_hook(i)))

    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                pils = _to_pil_list(images[start:start + batch_size])
                inputs = processor(images=pils, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model.vision_model(**inputs)
                logger.info(f"  CLIP layer features: {start + len(pils)}/{len(images)}")
    finally:
        for h in hooks:
            h.remove()
        del model

    return {i: torch.cat(bufs, dim=0).numpy() for i, bufs in layer_buffers.items()}


def extract_mae_layer_features(images: torch.Tensor, device: str,
                                 batch_size: int = 16) -> Dict[int, np.ndarray]:
    """
    Returns {layer_idx: (N, 768)} CLS-token array for each of the 12
    MAE ViT-Base encoder blocks.
    """
    from transformers import ViTMAEForPreTraining, AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base").to(device).eval()

    n_layers = len(model.vit.encoder.layer)
    logger.info(f"  MAE: {n_layers} encoder blocks")

    layer_buffers: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}

    hooks = []
    for i, block in enumerate(model.vit.encoder.layer):
        def _make_hook(idx):
            def _hook(module, input, output):
                # Newer transformers may return a tensor directly or a tuple
                hidden = output if isinstance(output, torch.Tensor) else output[0]
                cls = hidden[:, 0, :].float().cpu()
                layer_buffers[idx].append(cls)
            return _hook
        hooks.append(block.register_forward_hook(_make_hook(i)))

    try:
        with torch.no_grad():
            for start in range(0, len(images), batch_size):
                pils = _to_pil_list(images[start:start + batch_size])
                inputs = processor(images=pils, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model.vit(**inputs, noise=None)   # no masking
                logger.info(f"  MAE layer features: {start + len(pils)}/{len(images)}")
    finally:
        for h in hooks:
            h.remove()
        del model

    return {i: torch.cat(bufs, dim=0).numpy() for i, bufs in layer_buffers.items()}


# ── Probing loop ──────────────────────────────────────────────────────────────

def probe_all_layers(layer_feats: Dict[int, np.ndarray], labels: np.ndarray,
                     n_bootstrap: int = 500) -> Dict[int, Dict]:
    prober = LinearAccessibilityProber(test_size=0.2, random_state=42)
    results = {}
    for layer_idx in sorted(layer_feats.keys()):
        feats = layer_feats[layer_idx]
        logger.info(f"  Probing layer {layer_idx:2d}  features={feats.shape}")
        res = prober.probe(feats, labels, n_bootstrap=n_bootstrap)
        results[layer_idx] = res
        logger.info(
            f"    linear={res['linear_r2']:+.3f} [{res['linear_r2_ci'][0]:+.3f},{res['linear_r2_ci'][1]:+.3f}]"
            f"  Δ={res['delta_r2']:+.3f} [{res['delta_r2_ci'][0]:+.3f},{res['delta_r2_ci'][1]:+.3f}]"
        )
    return results


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_layer_results(all_results: Dict[str, Dict[int, Dict]], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    STYLE = {
        "CLIP ViT-B/32":  {"color": "#2980b9", "marker": "o", "ls": "-"},
        "MAE ViT-Base":   {"color": "#27ae60", "marker": "s", "ls": "-"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, layer_results in all_results.items():
        st = STYLE.get(model_name, {"color": "#7f8c8d", "marker": "x", "ls": "-"})
        layers = sorted(layer_results.keys())
        lin   = [layer_results[l]["linear_r2"]        for l in layers]
        lin_lo = [layer_results[l]["linear_r2_ci"][0] for l in layers]
        lin_hi = [layer_results[l]["linear_r2_ci"][1] for l in layers]
        dlt   = [layer_results[l]["delta_r2"]         for l in layers]
        dlt_lo = [layer_results[l]["delta_r2_ci"][0]  for l in layers]
        dlt_hi = [layer_results[l]["delta_r2_ci"][1]  for l in layers]

        x = np.array(layers)

        # Panel 1 — linear R²
        axes[0].plot(x, lin, color=st["color"], marker=st["marker"],
                     ls=st["ls"], lw=1.8, ms=6, label=model_name)
        axes[0].fill_between(x, lin_lo, lin_hi,
                              color=st["color"], alpha=0.15)

        # Panel 2 — Δ R²
        axes[1].plot(x, dlt, color=st["color"], marker=st["marker"],
                     ls=st["ls"], lw=1.8, ms=6, label=model_name)
        axes[1].fill_between(x, dlt_lo, dlt_hi,
                              color=st["color"], alpha=0.15)

    for ax, ylabel, title in [
        (axes[0], "Linear $R^2$",
         "Linear Accessibility vs. Encoder Depth\n(shaded = 95% bootstrap CI)"),
        (axes[1], "$\\Delta R^2$ (MLP $-$ Linear)",
         "Accessibility Gap vs. Encoder Depth\n(shaded = 95% bootstrap CI)"),
    ]:
        ax.set_xlabel("Transformer block (layer index)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(12))
        ax.axhline(0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = out_dir / f"layer_probing_figure.{ext}"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info(f"Saved: {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archs", nargs="+", default=["clip", "mae"],
                        choices=["clip", "mae"])
    parser.add_argument("--data-dir",   default="data/stimuli/")
    parser.add_argument("--output-dir", default="results/probing/")
    parser.add_argument("--max-stimuli", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "layer_probing_results.json"

    alphas = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    logger.info("Loading stimuli …")
    images, labels = load_all_stimuli(args.data_dir, alphas, args.max_stimuli)
    logger.info(f"  {len(images)} stimuli, labels range [{labels.min():.2f}, {labels.max():.2f}]")

    # Load any existing results so partial runs accumulate
    all_results: Dict[str, Dict] = {}
    if out_path.exists():
        with open(out_path) as f:
            raw = json.load(f)
        # Stored with string keys; convert back to int
        for model_name, layer_dict in raw.items():
            all_results[model_name] = {int(k): v for k, v in layer_dict.items()}
        logger.info(f"Loaded existing results: {list(all_results.keys())}")

    extractor_map = {
        "clip": ("CLIP ViT-B/32",  extract_clip_layer_features),
        "mae":  ("MAE ViT-Base",   extract_mae_layer_features),
    }

    for arch in args.archs:
        model_name, extractor = extractor_map[arch]
        if model_name in all_results:
            logger.info(f"Skipping {model_name} (already in results)")
            continue
        logger.info(f"\n=== {model_name} ===")
        layer_feats = extractor(images, device)
        logger.info(f"  Probing {len(layer_feats)} layers …")
        layer_results = probe_all_layers(layer_feats, labels, args.n_bootstrap)
        all_results[model_name] = layer_results

        # Save after each model so progress is never lost
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Saved intermediate results to {out_path}")

    logger.info("\nGenerating figure …")
    plot_layer_results(all_results, out_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
