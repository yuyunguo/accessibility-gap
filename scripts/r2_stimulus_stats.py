"""Round-2 Phase 1, items 1-3: stimulus-level statistics (no model features).

For every v2 stimulus file with nonzero α (α = 0 is not probed, as in the main analysis):

1. ``alpha_check``: the mean four-check product over adjacent 2×2 blocks of the texture
   grid (±1 coding, non-wrapping). This is the generator's ``realized_stats`` definition
   and the estimator printed in Figure 1. The wrap-around variant is also computed.
2. The same statistic at model-input scale after exact k-fold replication, checked
   against the identity  alpha_input = 1 - ((G-1)/(kG-1))**2 * (1 - alpha_check).
3. Nine pixel summaries (Supplement S1 definition: mean, variance, skewness, kurtosis,
   mean |horizontal| and |vertical| differences, horizontal, vertical, and diagonal
   neighbour products in ±1 coding) at model-input scale and at check scale.

Outputs: a JSON summary, and baseline feature files (``features``, ``labels``) that the
unchanged ``final_probe.py`` fits, so the baselines use the identical probe protocol.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from scipy import stats

logger = logging.getLogger(__name__)

ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)
# (subdirectory, set name, grid G, model input size). k = input / G.
CONDITIONS: Tuple[Tuple[str, str, int, int], ...] = (
    ("main_224", "main", 28, 224), ("main_224", "main", 56, 224), ("main_224", "main", 112, 224),
    ("main_224", "main", 32, 32),                      # ImageGPT: native 32×32 grid, k = 1
    ("main_sd", "main", 64, 512), ("main_sd", "main", 128, 512), ("main_sd", "main", 256, 512),
    ("replication", "rep", 56, 224), ("replication", "rep", 128, 512),
)
FIGURE_TILES = (0.1, 0.3, 0.5)                         # nonzero Figure 1 tiles: first map, main_224, G = 56


def load_maps(stim: Path, subdir: str, set_name: str, grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """All nonzero-α maps as (N, G, G) in ±1, in the order used by ``load_alpha_stimuli``."""
    imgs: List[np.ndarray] = []
    labels: List[float] = []
    for a in ALPHAS:
        maps = sio.loadmat(stim / subdir / f"alpha_{str(a).replace('.', 'p')}_{grid}x{grid}_{set_name}.mat")["maps"]
        maps = np.transpose(maps, (2, 0, 1)).astype(np.float64)
        if not set(np.unique(maps)) <= {0.0, 1.0}:
            raise ValueError(f"{subdir} α={a} G={grid}: stimulus is not binary")
        imgs.append(2.0 * maps - 1.0)
        labels += [a] * len(maps)
    return np.concatenate(imgs), np.asarray(labels)


def four_point(s: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Per-image mean product over adjacent 2×2 blocks of ±1 arrays (N, H, W)."""
    if wrap:
        r, c = np.roll(s, -1, axis=1), np.roll(s, -1, axis=2)
        return (s * r * c * np.roll(r, -1, axis=2)).mean(axis=(1, 2))
    return (s[:, :-1, :-1] * s[:, 1:, :-1] * s[:, :-1, 1:] * s[:, 1:, 1:]).mean(axis=(1, 2))


def replicate(s: np.ndarray, k: int) -> np.ndarray:
    """Exact k-fold pixel replication (identical to nearest interpolation by an integer factor)."""
    return s.repeat(k, axis=1).repeat(k, axis=2)


def nine_stats(s: np.ndarray) -> np.ndarray:
    """Nine pixel summaries per image (N, 9); intensities in [0, 1] for the moments."""
    x = (s + 1.0) / 2.0
    flat = x.reshape(len(x), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        skew = np.nan_to_num(stats.skew(flat, axis=1))
        kurt = np.nan_to_num(stats.kurtosis(flat, axis=1, fisher=True))
    return np.column_stack([
        flat.mean(1), flat.var(1), skew, kurt,
        np.abs(np.diff(x, axis=2)).mean(axis=(1, 2)), np.abs(np.diff(x, axis=1)).mean(axis=(1, 2)),
        (s[:, :, :-1] * s[:, :, 1:]).mean(axis=(1, 2)), (s[:, :-1, :] * s[:, 1:, :]).mean(axis=(1, 2)),
        (s[:, :-1, :-1] * s[:, 1:, 1:]).mean(axis=(1, 2)),
    ])


def describe(v: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Distribution of a per-image statistic at each nominal level."""
    out = {}
    for a in ALPHAS:
        x = v[labels == a]
        q = np.percentile(x, [2.5, 25, 50, 75, 97.5])
        out[str(a)] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "p2.5": float(q[0]),
                       "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p97.5": float(q[4])}
    return out


def analyse(stim: Path, subdir: str, set_name: str, grid: int, size: int, feat_dir: Path) -> Dict:
    s, y = load_maps(stim, subdir, set_name, grid)
    k = size // grid
    if k * grid != size:
        raise ValueError(f"G={grid} does not divide input {size}")
    a_chk, a_wrap = four_point(s), four_point(s, wrap=True)
    a_in_formula = 1.0 - ((grid - 1) / (k * grid - 1)) ** 2 * (1.0 - a_chk)
    nine_in, a_in = np.empty((len(s), 9)), np.empty(len(s))
    for i in range(0, len(s), 100):                    # chunked: 512² inputs are large
        x = replicate(s[i:i + 100], k)
        a_in[i:i + 100] = four_point(x)
        nine_in[i:i + 100] = nine_stats(x)
    nine_chk = nine_stats(s)
    tag = f"g{grid}_{set_name}"
    for name, feats in (("alphacheck", a_chk[:, None]), ("nine_input", nine_in), ("nine_check", nine_chk)):
        np.savez(feat_dir / f"base_{name}_{tag}.npz", features=feats, labels=y, alpha_check=a_chk)
    res = {"subdir": subdir, "set": set_name, "grid": grid, "input": size, "k": k, "n": int(len(y)),
           "identity_max_abs_err": float(np.abs(a_in - a_in_formula).max()),
           "alpha_input_mean_by_level": {str(a): float(a_in[y == a].mean()) for a in ALPHAS},
           "wrap_minus_nonwrap_mean_abs": float(np.abs(a_wrap - a_chk).mean()),
           "alpha_check": describe(a_chk, y),
           "corr_nominal_alpha_check": float(np.corrcoef(y, a_chk)[0, 1])}
    if (subdir, grid) == ("main_224", 56):
        first = {a: int(np.flatnonzero(y == a)[0]) for a in FIGURE_TILES}
        res["figure1_tiles"] = {str(a): {"alpha_check": float(a_chk[i]), "alpha_input": float(a_in[i])}
                                for a, i in first.items()}
    logger.info("%s: k=%d identity err %.2e; alpha_check SD at 0.3 = %.4f", tag, k,
                res["identity_max_abs_err"], res["alpha_check"]["0.3"]["sd"])
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stimuli", type=Path, required=True, help="data/stimuli_v2")
    parser.add_argument("--feat-dir", type=Path, required=True, help="output dir for baseline feature files")
    parser.add_argument("--out", type=Path, required=True, help="JSON summary")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.feat_dir.mkdir(parents=True, exist_ok=True)
    results = [analyse(args.stimuli, *c, args.feat_dir) for c in CONDITIONS]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"conditions": results}, indent=1))


if __name__ == "__main__":
    main()
