"""Figure 1: linear decodability of alpha and the nonlinear (RBF kernel) gap, from the frozen final run.

Reads only results/final/final_stats.json (commit 9001efd); computes nothing new.
  (A) ridge R² with 95% bootstrap CIs by check size k, per feature set.
  (B) kernel-minus-ridge gap with 95% CIs for all 35 controlled-texture conditions;
      filled markers = primary family (Holm-corrected), open = other conditions.

Run: python scripts/make_figure1.py --stats results/final/final_stats.json --out figures/fig1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# Feature-set key -> (label, colour). Okabe-Ito palette (colour-blind safe), plus Tol wine for ImageGPT.
MODELS: Dict[str, Tuple[str, str]] = {
    "clip": ("CLIP", "#0072B2"),
    "dino": ("DINO", "#E69F00"),
    "mae": ("MAE", "#009E73"),
    "resnet50": ("ResNet50", "#CC79A7"),
    "supvit_in1k": ("ViT-B/16, 1k", "#D55E00"),
    "supvit_21k": ("ViT-B/16, 21k", "#56B4E9"),
    "imagegpt": ("ImageGPT", "#882255"),   # not black: SD t=500 is dark grey
}
SD_SHADES = {100: "#A0A0A0", 300: "#666666", 500: "#222222"}
K_224 = {28: 8, 56: 4, 112: 2}          # grid size -> check size, 224-px models
K_SD = {64: 8, 128: 4, 256: 2}          # grid size -> check size, 512-px Stable Diffusion
K_ORDER = [8, 4, 2, 1]                  # x positions in panel A (coarse -> fine)


def parse(name: str) -> Dict:
    """Decode a condition key, e.g. 'dino_nearest_g28_main' or 'sd15_mid_t500_g128_rep'."""
    parts = name.split("_")
    grid = int(next(p for p in parts if p.startswith("g") and p[1:].isdigit())[1:])
    split = parts[-1]
    if name.startswith("sd15"):
        t = int(next(p for p in parts if p.startswith("t") and p[1:].isdigit())[1:])
        return {"model": "sd15", "t": t, "k": K_SD[grid], "split": split}
    model = name.split("_nearest_")[0]
    k = 1 if model == "imagegpt" else K_224[grid]
    return {"model": model, "t": None, "k": k, "split": split}


def load(path: Path) -> Tuple[List[Dict], List[str]]:
    d = json.loads(path.read_text())
    rows = []
    for name, v in d["conditions"].items():
        if name.startswith("nat_"):
            continue                      # natural textures are Table 5, not this figure
        rows.append({"name": name, **parse(name), **{k: v[k] for k in ("ridge", "ridge_ci", "gap", "ci_lo", "ci_hi")}})
    assert len(rows) == 35, f"expected 35 controlled-texture conditions, got {len(rows)}"
    assert len(d["primary_family"]) == 8 and all(p in d["conditions"] for p in d["primary_family"])
    return rows, d["primary_family"]


def panel_a(ax, rows: List[Dict]) -> None:
    main = [r for r in rows if r["split"] == "main"]
    xpos = {k: i for i, k in enumerate(K_ORDER)}
    series = [(m, lab, col, "-", "o") for m, (lab, col) in MODELS.items()]
    series += [(f"sd15_t{t}", f"SD, t = {t}", col, "--", "s") for t, col in SD_SHADES.items()]
    for key, lab, col, ls, mk in series:
        pts = [r for r in main if (r["model"] == key if not key.startswith("sd15")
                                   else r["model"] == "sd15" and f"t{r['t']}" == key[5:])]
        pts.sort(key=lambda r: xpos[r["k"]])
        x = [xpos[r["k"]] for r in pts]
        y = [r["ridge"] for r in pts]
        err = [[r["ridge"] - r["ridge_ci"][0] for r in pts], [r["ridge_ci"][1] - r["ridge"] for r in pts]]
        ax.errorbar(x, y, yerr=err, color=col, ls=ls, marker=mk, ms=3.5, lw=1, capsize=1.5, label=lab)
    ax.set_xticks(range(len(K_ORDER)), [f"{k}" for k in K_ORDER])
    ax.set_xlabel("Check size $k$ (pixels per texture pixel)")
    ax.set_ylabel(r"Ridge $R^2_{\mathrm{lin}}$ (test)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))  # below axes: SD t=500 occupies the lower left
    ax.set_title("A  Linear decodability of $\\alpha$", loc="left", fontsize=8, fontweight="bold")


def label(r: Dict) -> str:
    base = "SD mid, t = %d" % r["t"] if r["model"] == "sd15" else MODELS[r["model"]][0]
    rep = ", rep." if r["split"] == "rep" else ""
    return f"{base}, k = {r['k']}{rep}"


def panel_b(ax, rows: List[Dict], primary: List[str]) -> None:
    order = list(MODELS) + ["sd15"]
    rows = sorted(rows, key=lambda r: (order.index(r["model"]), r["t"] or 0, -r["k"], r["split"] != "main"))
    for i, r in enumerate(rows):
        col = SD_SHADES[r["t"]] if r["model"] == "sd15" else MODELS[r["model"]][1]
        is_primary = r["name"] in primary
        ax.errorbar(r["gap"], i, xerr=[[r["gap"] - r["ci_lo"]], [r["ci_hi"] - r["gap"]]], color=col,
                    marker="o", ms=3.5, mfc=col if is_primary else "white", lw=1, capsize=1.5)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=6)
    ax.set_xticks([-0.004, 0.0, 0.004, 0.008])   # denser ticks collide at 5.4-in width
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta R^2$ = kernel $-$ ridge (95% CI)")
    ax.set_title("B  Nonlinear advantage", loc="left", fontsize=8, fontweight="bold")
    ax.text(0.98, 0.01, "filled: primary family", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=Path("results/final/final_stats.json"))
    ap.add_argument("--out", type=Path, default=Path("figures/fig1"))
    ap.add_argument("--width", type=float, default=5.4, help="inches; 5.4 = elsarticle preprint text width (390 pt)")
    ap.add_argument("--height", type=float, default=5.6, help="inches")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)   # PDF font subsetting is chatty at INFO

    rows, primary = load(args.stats)
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig, (a, b) = plt.subplots(1, 2, figsize=(args.width, args.height), gridspec_kw={"width_ratios": [1, 1.25]})
    panel_a(a, rows)
    panel_b(b, rows, primary)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(args.out.with_suffix(f".{ext}"), bbox_inches="tight", **kw)
    gaps = [r["gap"] for r in rows]
    logger.info("35 conditions; gap %.4f..%.4f; max CI upper %.4f; wrote %s.{pdf,png}",
                min(gaps), max(gaps), max(r["ci_hi"] for r in rows), args.out)


if __name__ == "__main__":
    main()
