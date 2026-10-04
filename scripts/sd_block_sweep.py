"""Extraction-location sweep for the SD v1.5 UNet (reviewer R2-1 / R1-5).

Hooks every down block, the mid block, and every up block, spatially mean-pools
each output, and saves one feature file per (block, timestep, seed). Inputs are
v2 seeded stimuli magnified to 512 by exact integer replication; noise is seeded
exactly as in extract_sd_seeded.py, so mid-block features match that script.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline

from extract_sd_seeded import encode_latents, load_alpha_stimuli

logger = logging.getLogger(__name__)


def block_modules(unet) -> Dict[str, torch.nn.Module]:
    mods = {f"down{i}": b for i, b in enumerate(unet.down_blocks)}
    mods["mid"] = unet.mid_block
    mods.update({f"up{i}": b for i, b in enumerate(unet.up_blocks)})
    return mods


@torch.no_grad()
def sweep(unet, scheduler, latents, noise, t, null_emb, device, dtype, batch) -> Dict[str, np.ndarray]:
    mods = block_modules(unet)
    cur: Dict[str, torch.Tensor] = {}
    handles = []
    for name, m in mods.items():
        def hook(_m, _i, out, name=name):
            h = out[0] if isinstance(out, (tuple, list)) else out
            cur[name] = F.adaptive_avg_pool2d(h, 1).flatten(1).float().cpu()
        handles.append(m.register_forward_hook(hook))
    feats: Dict[str, List[torch.Tensor]] = {n: [] for n in mods}
    try:
        for i in range(0, len(latents), batch):
            z0, eps = latents[i:i + batch].to(device), noise[i:i + batch].to(device)
            tt = torch.full((len(z0),), t, device=device, dtype=torch.long)
            zt = scheduler.add_noise(z0, eps, tt).to(dtype)
            unet(zt, tt, encoder_hidden_states=null_emb.expand(len(z0), -1, -1).to(dtype))
            for n in mods:
                feats[n].append(cur[n])
    finally:
        for h in handles:
            h.remove()
    return {n: torch.cat(v).numpy() for n, v in feats.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stim-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--set-name", default="main")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 300, 500])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    device, dtype = "cuda", torch.float16
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=dtype,
                                                   safety_checker=None, requires_safety_checker=False)
    vae, unet, scheduler = pipe.vae.to(device).eval(), pipe.unet.to(device).eval(), pipe.scheduler
    ids = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         return_tensors="pt").input_ids
    null_emb = pipe.text_encoder(ids)[0].detach().to(device)

    images, labels = load_alpha_stimuli(args.stim_dir, args.grid, args.set_name)
    if 512 % images.shape[-1] != 0:
        raise ValueError("grid must divide 512 for exact replication")
    latents = encode_latents(vae, images, device, dtype, args.batch, "nearest")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        noise = torch.randn(latents.shape, generator=torch.Generator(device="cpu").manual_seed(s))
        for t in args.timesteps:
            for block, f in sweep(unet, scheduler, latents, noise, t, null_emb, device, dtype, args.batch).items():
                np.savez(args.out_dir / f"sd15_{block}_t{t}_s{s}_g{args.grid}_{args.set_name}.npz",
                         features=f, labels=labels, block=block, timestep=t, seed=s)
            logger.info(f"seed={s} t={t}: saved {len(unet.down_blocks) * 2 + 1} blocks")


if __name__ == "__main__":
    main()
