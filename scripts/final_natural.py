"""Final analysis: natural-texture (DTD) features with the corrected extractors.

Uses the 5,640 DTD 64×64 grayscale patches and their four-point statistic γ from
the legacy pilot (``dtd_patches_gamma.npz``). γ is first recomputed from the
patches with the legacy definition and must match the stored values exactly.
Features come from the same corrected code paths as the BTC analysis:
unmasked MAE, ImageNet-1k supervised ViT (plus the 21k checkpoint), CLIP, DINO,
ResNet50, ImageGPT (centre 32×32 crop), and SD v1.5 mid-block (64→512 exact
replication; seed 0; t = 100, 300, 500). These are natural images, so no
isolation of γ from lower-order statistics is claimed.
"""

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from extract_features_v2 import extract_imagegpt, extract_standard
from extract_sd_seeded import encode_latents, midblock_features

logger = logging.getLogger(__name__)
STANDARD = ["clip", "mae", "dino", "supvit_in1k", "supvit_21k", "resnet50"]


def gamma_legacy(patch: np.ndarray) -> float:
    """Legacy pilot definition: binarize at the patch median, mean 2×2 four-point product."""
    b = np.where(patch >= np.median(patch), 1.0, -1.0)
    return float(np.mean(b[:-1, :-1] * b[1:, :-1] * b[:-1, 1:] * b[1:, 1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--patches", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sd-timesteps", type=int, nargs="+", default=[100, 300, 500])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    d = np.load(args.patches)
    patches, gammas = d["patches"].astype(np.float32), d["gammas"].astype(np.float64)
    recomputed = np.array([gamma_legacy(p) for p in patches])
    max_dev = float(np.abs(recomputed - gammas).max())
    logger.info(f"gamma recomputation: max |diff| = {max_dev:.3g} over {len(patches)} patches")
    assert max_dev < 1e-6, "stored gammas do not match the legacy definition"
    images = torch.from_numpy(patches)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    def save(name: str, f: np.ndarray) -> None:
        np.savez(args.out_dir / f"nat_{name}.npz", features=f, gammas=gammas)
        logger.info(f"nat_{name}: {f.shape}")

    for name in STANDARD:      # 64 → 224 nearest (non-integer; natural images, no isolation claim)
        save(name, extract_standard(name, images, "nearest", "cuda", 32))
        torch.cuda.empty_cache()
    save("imagegpt", extract_imagegpt(images, "nearest", "cuda", 8))

    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,
                                                   safety_checker=None, requires_safety_checker=False)
    vae, unet = pipe.vae.to("cuda").eval(), pipe.unet.to("cuda").eval()
    ids = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         return_tensors="pt").input_ids
    null_emb = pipe.text_encoder(ids)[0].detach().to("cuda")
    latents = encode_latents(vae, images, "cuda", torch.float16, 16, "nearest")   # 64 → 512 exact (8×)
    noise = torch.randn(latents.shape, generator=torch.Generator(device="cpu").manual_seed(0))
    for t in args.sd_timesteps:
        save(f"sd15_mid_t{t}_s0", midblock_features(unet, pipe.scheduler, latents, noise, t, null_emb,
                                                   "cuda", torch.float16, 16))


if __name__ == "__main__":
    main()
