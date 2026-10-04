"""Feature extraction with explicit, auditable preprocessing (revision v2).

Every model receives the binary stimulus through a preprocessing path we control
(no hidden processor resizing). ``--resize nearest`` keeps the input binary; for
v2 stimuli (``--set-name`` other than ``legacy``) the magnification must be an
integer, so it is exact pixel replication and every stimulus statistic is preserved.
``--resize bilinear`` approximates the submitted pipeline: it is identical for DINO,
the supervised ViTs and ResNet50 (verified, row-wise r = 1.000), but CLIP's original
processor used bicubic resizing (row-wise r = 0.79 against the cached features).

Fixes relative to the legacy code:
  * MAE runs with ``mask_ratio=0`` (legacy masked 75% of patches at random).
  * The supervised ViT is available as the ImageNet-1k-only timm checkpoint
    (``supvit_in1k``) and the legacy 21k→1k checkpoint (``supvit_21k``).
  * ImageGPT sees a native-scale 32×32 centre crop (``--resize nearest``) instead
    of a bilinear 64→32 downsample that turns α into a first-order cue.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F

from extract_sd_seeded import load_alpha_stimuli

logger = logging.getLogger(__name__)

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
CLIP_NORM = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def to_rgb(images: torch.Tensor, size: int, mode: str, exact: bool = False) -> torch.Tensor:
    """(N, g, g) in [0, 1] -> (N, 3, size, size). ``exact`` requires integer magnification."""
    if exact and size % images.shape[-1] != 0:
        raise ValueError(f"grid {images.shape[-1]} does not divide input size {size}: not exact replication")
    x = images.unsqueeze(1).repeat(1, 3, 1, 1)
    if mode == "nearest":
        return F.interpolate(x, size=(size, size), mode="nearest")
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def normalize(x: torch.Tensor, norm: Tuple) -> torch.Tensor:
    mean = torch.tensor(norm[0]).view(1, 3, 1, 1)
    std = torch.tensor(norm[1]).view(1, 3, 1, 1)
    return (x - mean) / std


def build_model(name: str, device: str) -> Tuple[Callable[[torch.Tensor], torch.Tensor], int, Tuple]:
    """Return (forward_fn mapping a normalized batch to features, input size, normalization)."""
    if name == "clip":
        from transformers import CLIPVisionModel
        m = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        return (lambda x: m(pixel_values=x).pooler_output), 224, CLIP_NORM
    if name == "mae":
        from transformers import ViTMAEModel
        m = ViTMAEModel.from_pretrained("facebook/vit-mae-base", mask_ratio=0.0).to(device).eval()
        return (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), 224, IMAGENET
    if name == "dino":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("facebook/dino-vitb16").to(device).eval()
        return (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), 224, IMAGENET
    if name == "supvit_21k":
        from transformers import ViTModel
        m = ViTModel.from_pretrained("google/vit-base-patch16-224", add_pooling_layer=False).to(device).eval()
        return (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), 224, HALF
    if name == "supvit_in1k":
        import timm
        m = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=True).to(device).eval()
        return (lambda x: m.forward_head(m.forward_features(x), pre_logits=True)), 224, HALF
    if name == "resnet50":
        import torchvision.models as tvm
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
        m.fc = torch.nn.Identity()
        return (lambda x: m(x)), 224, IMAGENET
    raise ValueError(f"unknown model {name}")


@torch.no_grad()
def extract_standard(name: str, images: torch.Tensor, mode: str, device: str, batch: int,
                     exact: bool = False) -> np.ndarray:
    fwd, size, norm = build_model(name, device)
    feats = []
    for i in range(0, len(images), batch):
        x = normalize(to_rgb(images[i:i + batch], size, mode, exact), norm).to(device)
        feats.append(fwd(x).float().cpu())
    return torch.cat(feats).numpy()


@torch.no_grad()
def extract_imagegpt(images: torch.Tensor, mode: str, device: str, batch: int) -> np.ndarray:
    """Mean final-layer hidden state of ImageGPT-medium (legacy feature definition)."""
    from transformers import ImageGPTImageProcessor, ImageGPTModel
    proc = ImageGPTImageProcessor.from_pretrained("openai/imagegpt-medium")
    m = ImageGPTModel.from_pretrained("openai/imagegpt-medium").to(device).eval()
    if mode == "nearest":   # native scale: full 32×32 grid, or a 32×32 centre crop of a larger grid
        g = images.shape[-1]
        o = (g - 32) // 2
        if g < 32 and 32 % g != 0:
            raise ValueError(f"grid {g} cannot be replicated exactly to 32")
        crops = images[:, o:o + 32, o:o + 32] if g >= 32 else F.interpolate(
            images.unsqueeze(1), size=(32, 32), mode="nearest")[:, 0]
    else:                   # legacy: bilinear 64→32
        crops = F.interpolate(images.unsqueeze(1), size=(32, 32), mode="bilinear", align_corners=False)[:, 0]
    rgb = (crops.unsqueeze(-1).repeat(1, 1, 1, 3) * 255).round().to(torch.uint8).numpy()
    feats = []
    for i in range(0, len(rgb), batch):
        ids = proc(images=list(rgb[i:i + batch]), do_resize=False, return_tensors="pt")["input_ids"].to(device)
        feats.append(m(input_ids=ids).last_hidden_state.mean(dim=1).float().cpu())
    return torch.cat(feats).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stim-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resize", choices=["nearest", "bilinear"], required=True)
    parser.add_argument("--models", nargs="+",
                        default=["clip", "mae", "dino", "supvit_in1k", "supvit_21k", "resnet50", "imagegpt"])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--grid", type=int, default=64, help="texture grid size of the stimulus files")
    parser.add_argument("--set-name", default="legacy", help="'legacy' or the v2 set name (e.g. main, rep)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    images, labels = load_alpha_stimuli(args.stim_dir, args.grid, args.set_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        f = (extract_imagegpt(images, args.resize, "cuda", 8) if name == "imagegpt"
             else extract_standard(name, images, args.resize, "cuda", args.batch,
                                   exact=(args.resize == "nearest" and args.set_name != "legacy")))
        tag = "" if args.set_name == "legacy" else f"_g{args.grid}_{args.set_name}"
        np.savez(args.out_dir / f"{name}_{args.resize}{tag}.npz", features=f, labels=labels,
                 resize=args.resize, grid=args.grid, set_name=args.set_name)
        logger.info(f"{name} [{args.resize}]: features {f.shape}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
