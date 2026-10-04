"""Seeded Stable Diffusion v1.5 UNet mid-block feature extraction.

Preprocessing matches the legacy analyzer (3-channel replication, bilinear 64→512,
[-1, 1] scaling, VAE posterior mean × scaling_factor, null-text conditioning,
spatial mean pooling of the mid-block output). Unlike the legacy code, forward
noise is drawn from a seeded ``torch.Generator``: for a given seed the same noise
tensor is used at every timestep (common random numbers across t).
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


class _StopForward(Exception):
    """Raised by the mid-block hook to skip the UNet decoder."""


def _stim_path(stim_dir: Path, a: float, grid: int, set_name: str) -> Path:
    """Legacy files (64×64, shared seeds) or v2 seeded files from generate_btc_octave.py."""
    if set_name == "legacy":
        return stim_dir / f"informative_alpha_{str(a).replace('.', '')}_64x64.mat"
    return stim_dir / f"alpha_{str(a).replace('.', 'p')}_{grid}x{grid}_{set_name}.mat"


def load_alpha_stimuli(stim_dir: Path, grid: int = 64, set_name: str = "legacy") -> Tuple[torch.Tensor, np.ndarray]:
    """Load the six nonzero-α stimulus files as (N, grid, grid) in [0, 1] with labels."""
    images: List[np.ndarray] = []
    labels: List[float] = []
    for a in ALPHAS:
        maps = sio.loadmat(_stim_path(stim_dir, a, grid, set_name))["maps"]
        maps = np.transpose(maps, (2, 0, 1)) if maps.ndim == 3 else maps[np.newaxis]
        maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-8)
        images.append(maps.astype(np.float32))
        labels += [a] * len(maps)
    return torch.from_numpy(np.concatenate(images)), np.asarray(labels)


@torch.no_grad()
def encode_latents(vae, images: torch.Tensor, device: str, dtype, batch: int, resize: str = "bilinear") -> torch.Tensor:
    """VAE posterior mean × scaling factor for every stimulus (deterministic)."""
    out = []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].unsqueeze(1).repeat(1, 3, 1, 1)
        x = (F.interpolate(x, size=(512, 512), mode="nearest") if resize == "nearest"
             else F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False))
        x = (2.0 * x - 1.0).to(device, dtype)
        out.append(vae.encode(x).latent_dist.mean * vae.config.scaling_factor)
    return torch.cat(out).float().cpu()


@torch.no_grad()
def midblock_features(unet, scheduler, latents: torch.Tensor, noise: torch.Tensor, t: int,
                      null_emb: torch.Tensor, device: str, dtype, batch: int) -> np.ndarray:
    """Pooled mid-block activations for noisy latents at timestep ``t``."""
    captured = {}

    def hook(_m, _i, out):
        captured["f"] = out[0] if isinstance(out, (tuple, list)) else out
        raise _StopForward

    handle = unet.mid_block.register_forward_hook(hook)
    feats = []
    try:
        for i in range(0, len(latents), batch):
            z0 = latents[i:i + batch].to(device)
            eps = noise[i:i + batch].to(device)
            tt = torch.full((len(z0),), t, device=device, dtype=torch.long)
            zt = scheduler.add_noise(z0, eps, tt).to(dtype)
            try:
                unet(zt, tt, encoder_hidden_states=null_emb.expand(len(z0), -1, -1).to(dtype))
            except _StopForward:
                pass
            feats.append(F.adaptive_avg_pool2d(captured["f"], 1).flatten(1).float().cpu())
    finally:
        handle.remove()
    return torch.cat(feats).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stim-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 300, 500, 700, 900])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--resize", choices=["bilinear", "nearest"], default="bilinear")
    parser.add_argument("--grid", type=int, default=64, help="texture grid size of the stimulus files")
    parser.add_argument("--set-name", default="legacy", help="'legacy' or the v2 set name (e.g. main, rep)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device, dtype = "cuda", torch.float16
    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None,
                                                   requires_safety_checker=False)
    vae, unet, scheduler = pipe.vae.to(device).eval(), pipe.unet.to(device).eval(), pipe.scheduler
    ids = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         return_tensors="pt").input_ids
    null_emb = pipe.text_encoder(ids)[0].detach().to(device)

    images, labels = load_alpha_stimuli(args.stim_dir, args.grid, args.set_name)
    if args.resize == "nearest" and args.set_name != "legacy" and 512 % images.shape[-1] != 0:
        raise ValueError(f"grid {images.shape[-1]} does not divide 512: not exact replication")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    latents = encode_latents(vae, images, device, dtype, args.batch, args.resize)
    np.savez(args.out_dir / f"vae_latents_{args.resize}_g{images.shape[-1]}_{args.set_name}.npz",
             latents=latents.numpy(), labels=labels)
    logger.info(f"Encoded {len(latents)} latents {tuple(latents.shape[1:])}")

    for s in args.seeds:
        gen = torch.Generator(device="cpu").manual_seed(s)
        noise = torch.randn(latents.shape, generator=gen)
        for t in args.timesteps:
            f = midblock_features(unet, scheduler, latents, noise, t, null_emb, device, dtype, args.batch)
            tag = "" if args.resize == "bilinear" else f"_{args.resize}"
            if args.set_name != "legacy":
                tag += f"_g{args.grid}_{args.set_name}"
            np.savez(args.out_dir / f"sd15_mid_t{t}_s{s}{tag}.npz", features=f, labels=labels,
                     timestep=t, seed=s, resize=args.resize)
            logger.info(f"seed={s} t={t}: features {f.shape}")


if __name__ == "__main__":
    main()
