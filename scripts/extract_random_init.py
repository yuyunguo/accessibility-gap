"""Round-2 Phase 2: features from randomly initialized (untrained) networks.

Each architecture is built from its pretrained counterpart's config at the library's
default initialization (seeded per init seed) and read out exactly like the counterpart
in ``extract_features_v2.build_model`` / ``extract_sd_seeded``. Stimuli, exact
replication, normalization, pooling, and file layout are unchanged, so ``final_probe.py``
and ``final_stats.py`` run on the outputs as they are.

Models (counterpart in brackets):
  rand_vit_hf_imnorm / rand_vit_hf_halfnorm  HF ViT-B/16, CLS token  [MAE, DINO / ViT-21k]
  rand_vit_timm                              timm ViT-B/16, pre-logits  [ViT-1k]
  rand_clip                                  CLIP ViT-B/32 vision tower, pooler_output  [CLIP]
  rand_resnet50                              torchvision ResNet50, pooled  [ResNet50]
  rand_relu                                  ReLU(W x + b) of the 224-px input, 768-d  [none]
  sd15 (--sd)                                random UNet (pretrained VAE unless --random-vae)
Random modules run in float32 (an untrained fp16 network can overflow); pretrained SD
components keep the float16 used in the main analysis.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch

from extract_features_v2 import CLIP_NORM, HALF, IMAGENET, normalize, to_rgb
from extract_sd_seeded import encode_latents, load_alpha_stimuli, midblock_features

logger = logging.getLogger(__name__)
STANDARD = ("rand_vit_hf_imnorm", "rand_vit_hf_halfnorm", "rand_vit_timm", "rand_clip", "rand_resnet50", "rand_relu")


def build_random(name: str, init_seed: int, device: str) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Tuple]:
    """(forward_fn on a normalized (N, 3, 224, 224) batch, normalization) for a random network."""
    torch.manual_seed(init_seed)
    if name.startswith("rand_vit_hf"):
        from transformers import ViTConfig, ViTModel
        m = ViTModel(ViTConfig.from_pretrained("google/vit-base-patch16-224"), add_pooling_layer=False)
        fwd, norm = (lambda x: m(pixel_values=x).last_hidden_state[:, 0]), (IMAGENET if "imnorm" in name else HALF)
    elif name == "rand_vit_timm":
        import timm
        m = timm.create_model("vit_base_patch16_224.augreg_in1k", pretrained=False)
        fwd, norm = (lambda x: m.forward_head(m.forward_features(x), pre_logits=True)), HALF
    elif name == "rand_clip":
        from transformers import CLIPVisionConfig, CLIPVisionModel
        m = CLIPVisionModel(CLIPVisionConfig.from_pretrained("openai/clip-vit-base-patch32"))
        fwd, norm = (lambda x: m(pixel_values=x).pooler_output), CLIP_NORM
    elif name == "rand_resnet50":
        import torchvision.models as tvm
        m = tvm.resnet50(weights=None)
        m.fc = torch.nn.Identity()
        fwd, norm = (lambda x: m(x)), IMAGENET
    elif name == "rand_relu":
        g = torch.Generator().manual_seed(init_seed)
        d_in = 224 * 224
        W = (torch.randn(d_in, 768, generator=g) / d_in ** 0.5).to(device)
        b = torch.randn(768, generator=g).to(device)
        # Input: one channel of the replicated image in ±1 coding (normalization undone below).
        return (lambda x: torch.relu((2.0 * x[:, 0].flatten(1) - 1.0) @ W + b)), ((0.0,) * 3, (1.0,) * 3)
    else:
        raise ValueError(f"unknown model {name}")
    m.to(device).eval()
    return fwd, norm


@torch.no_grad()
def extract(name: str, init_seed: int, images: torch.Tensor, device: str, batch: int) -> np.ndarray:
    fwd, norm = build_random(name, init_seed, device)
    feats = []
    for i in range(0, len(images), batch):
        x = normalize(to_rgb(images[i:i + batch], 224, "nearest", exact=True), norm).to(device)
        feats.append(fwd(x).float().cpu())
    f = torch.cat(feats).numpy()
    if not np.isfinite(f).all():
        raise FloatingPointError(f"{name} init {init_seed}: non-finite features")
    return f


def run_sd(args: argparse.Namespace, device: str) -> None:
    from diffusers import StableDiffusionPipeline, UNet2DConditionModel, AutoencoderKL
    pipe = StableDiffusionPipeline.from_pretrained(args.sd_model, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    ids = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         return_tensors="pt").input_ids
    null_emb = pipe.text_encoder(ids)[0].detach().to(device).float()
    images, labels = load_alpha_stimuli(args.stimuli / "main_sd", args.sd_grid, "main")
    for i in args.init_seeds:
        torch.manual_seed(i)
        unet = UNet2DConditionModel.from_config(pipe.unet.config).to(device).eval()
        if args.random_vae:
            torch.manual_seed(10_000 + i)
            vae, vdt = AutoencoderKL.from_config(pipe.vae.config).to(device).eval(), torch.float32
        else:
            vae, vdt = pipe.vae.to(device).eval(), torch.float16
        latents = encode_latents(vae, images, device, vdt, args.batch, "nearest")
        tag = f"rinit{i}" + ("v" if args.random_vae else "")
        for s in args.noise_seeds:
            noise = torch.randn(latents.shape, generator=torch.Generator(device="cpu").manual_seed(s))
            for t in args.timesteps:
                f = midblock_features(unet, pipe.scheduler, latents, noise, t, null_emb, device, torch.float32,
                                      args.batch)
                if not np.isfinite(f).all():
                    raise FloatingPointError(f"SD {tag} s{s} t{t}: non-finite features")
                np.savez(args.out_dir / f"sd15_mid_t{t}_s{s}_nearest_g{args.sd_grid}_{tag}.npz", features=f,
                         labels=labels, timestep=t, seed=s, init_seed=i, random_vae=args.random_vae)
                logger.info("SD %s noise=%d t=%d: %s", tag, s, t, f.shape)
        del unet
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stimuli", type=Path, required=True, help="data/stimuli_v2")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(STANDARD), choices=STANDARD)
    parser.add_argument("--grids", type=int, nargs="+", default=[28, 56, 112])
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--sd", action="store_true", help="run the random-UNet SD extraction instead")
    parser.add_argument("--random-vae", action="store_true", help="SD: randomize the VAE as well")
    parser.add_argument("--sd-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--sd-grid", type=int, default=128)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 500])
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sd:
        run_sd(args, "cuda")
        return
    for g in args.grids:
        images, labels = load_alpha_stimuli(args.stimuli / "main_224", g, "main")
        for name in args.models:
            for i in args.init_seeds:
                f = extract(name, i, images, "cuda", args.batch)
                np.savez(args.out_dir / f"{name}_s{i}_nearest_g{g}_main.npz", features=f, labels=labels,
                         init_seed=i, grid=g)
                logger.info("%s init=%d g=%d: %s", name, i, g, f.shape)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
