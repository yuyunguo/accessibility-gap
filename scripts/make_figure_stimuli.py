"""Stimulus figure: example BTC textures and what the model receives under each resizing pipeline.

  (A) First texture of each main-set file at k = 4 (56×56 grid -> 224×224 by exact replication),
      for alpha = 0 (reference: validated, not probed), 0.1, 0.3, 0.5. Each tile is annotated with
      the alpha estimated from that single image (mean product over all 2×2 blocks, ±1 coding).
  (B) Zoomed crop of the 224×224 model input: this study (56×56 grid, ×4 exact replication) vs the
      earlier analysis's resizing (64×64 grid, bilinear to 224, align_corners=False), both at
      alpha = 0.3 and both from the new stimulus files. Resizing uses the pipeline's own to_rgb.
  (C) Pixel-value histograms of the two full 224×224 inputs.

Run: python scripts/make_figure_stimuli.py --stimuli <data/stimuli_v2> --out figures/fig_stimuli
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import torch  # noqa: E402

from extract_features_v2 import to_rgb  # noqa: E402  (the exact resizing used for feature extraction)

logger = logging.getLogger(__name__)

TILE_ALPHAS = (0.0, 0.1, 0.3, 0.5)
CROP_ALPHA = 0.3
CROP = 48                      # crop side in model-input pixels (224×224 input)
INPUT = 224


def tag(a: float) -> str:
    return str(a).replace(".", "p")


def load_map(stim: Path, subdir: str, alpha: float, grid: int) -> Tuple[np.ndarray, Dict]:
    """First texture of a stimulus file as float32 in {0, 1}, plus its manifest entry."""
    name = f"alpha_{tag(alpha)}_{grid}x{grid}_main.mat"
    maps = sio.loadmat(stim / subdir / name)["maps"]
    img = maps[:, :, 0].astype(np.float32)
    assert set(np.unique(img)) <= {0.0, 1.0}, f"{name}: stimulus is not binary"
    entry = next(e for e in json.loads((stim / subdir / "manifest.json").read_text()) if e["file"] == name)
    return img, entry


def image_alpha(img: np.ndarray) -> float:
    """Alpha of one image: mean over all (non-wrapping) 2×2 blocks of the product of ±1 pixels."""
    s = 2.0 * img - 1.0
    return float(np.mean(s[:-1, :-1] * s[1:, :-1] * s[:-1, 1:] * s[1:, 1:]))


def model_input(img: np.ndarray, mode: str) -> np.ndarray:
    """One grey channel of the 224×224 model input produced by the pipeline's to_rgb."""
    x = to_rgb(torch.from_numpy(img)[None], INPUT, mode, exact=(mode == "nearest"))
    return x[0, 0].numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stimuli", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("figures/fig_stimuli"))
    ap.add_argument("--width", type=float, default=6.5, help="inches; manuscript text width")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})

    fig = plt.figure(figsize=(args.width, 3.7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.6, wspace=0.25)
    bottom = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 2.1], wspace=0.55)

    # (A) example textures at k = 4, as presented to the 224-px models
    for i, a in enumerate(TILE_ALPHAS):
        img, entry = load_map(args.stimuli, "main_224", a, 56)
        x = model_input(img, "nearest")
        assert set(np.unique(x)) <= {0.0, 1.0}, "exact replication must stay binary"
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(x, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ref = " (reference)" if a == 0 else ""
        ax.set_title(f"$\\alpha$ = {a:g}{ref}", fontsize=7)
        est = image_alpha(img)
        ax.set_xlabel(f"image $\\hat{{\\alpha}}$ = {0.0 if abs(est) < 0.005 else est:+.2f}", fontsize=6)
        logger.info("%s seed=%d file-level realized alpha=%+.4f image alpha=%+.4f",
                    entry["file"], entry["seed"], entry["realized"]["alpha"], image_alpha(img))
        if i == 0:
            ax.text(-0.12, 1.12, "A", transform=ax.transAxes, fontsize=9, fontweight="bold")

    # (B) zoomed crops of the model input under the two pipelines
    new_img, new_e = load_map(args.stimuli, "main_224", CROP_ALPHA, 56)
    old_img, old_e = load_map(args.stimuli, "main_sd", CROP_ALPHA, 64)
    new_x, old_x = model_input(new_img, "nearest"), model_input(old_img, "bilinear")
    for j, (x, title) in enumerate(((new_x, "This study\n56×56 grid, ×4 replication"),
                                    (old_x, "Earlier analysis\n64×64 grid, bilinear"))):
        ax = fig.add_subplot(bottom[0, j])
        ax.imshow(x[:CROP, :CROP], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=6.5)
        if j == 0:
            ax.text(-0.12, 1.32, "B", transform=ax.transAxes, fontsize=9, fontweight="bold")

    # (C) pixel-value histograms of the full 224×224 inputs
    ax = fig.add_subplot(bottom[0, 2])
    bins = np.linspace(0, 1, 21)
    for x, lab, col in ((new_x, "this study", "#0072B2"), (old_x, "earlier analysis (bilinear)", "#D55E00")):
        frac = float(np.mean((x > 0) & (x < 1)))
        ax.hist(x.ravel(), bins=bins, weights=np.full(x.size, 1 / x.size), histtype="step", lw=1.2,
                color=col, label=f"{lab}: {100 * frac:.0f}% intermediate")
        logger.info("%s: fraction of pixels strictly between 0 and 1 = %.4f", lab, frac)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1)
    ax.set_xlabel("Pixel value at model input")
    ax.set_ylabel("Fraction of pixels")
    ax.legend(fontsize=6, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))  # above axes: bars fill the plot
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.2, 1.32, "C", transform=ax.transAxes, fontsize=9, fontweight="bold")
    logger.info("crops: %s (seed %d), %s (seed %d)", new_e["file"], new_e["seed"], old_e["file"], old_e["seed"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(args.out.with_suffix(f".{ext}"), bbox_inches="tight", **kw)
    logger.info("wrote %s.{pdf,png}", args.out)


if __name__ == "__main__":
    main()
