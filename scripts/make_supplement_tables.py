"""Print the supplement cells that are filled from workstation result files (read-only; computes nothing new).

  S2  ridge train R² for the primary family   <- results/final/oof/oof_*.json  (mean over folds; SD: over seeds too)
  S4  ridge R² by UNet stage at t = 100, 300  <- results/confirm/audit_blocks_*.json  (range over seeds 0-1)
      and the largest permutation-null R²    <- same files ("*_permuted" summaries)
  S6  latent-channel-mean readout, t = 100/300 <- results/confirm/latent_probe_*.json  (range over seeds 0-1)
  S2  audit probes (k-NN, boosting, MLPs)     <- results/confirm/audit_v3_*.json  (all summaries printed for review)
  S5  the 20 reproduced values               <- results/reproduction_baseline.json  (structure printed for review)

Run on the workstation from the repo root: python scripts/make_supplement_tables.py
"""

import glob
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)
ROOT = Path("results")
PRIMARY = ["clip_nearest_g56_main", "dino_nearest_g56_main", "mae_nearest_g56_main", "resnet50_nearest_g56_main",
           "supvit_in1k_nearest_g56_main", "supvit_21k_nearest_g56_main", "imagegpt_nearest_g32_main"]
STAGES = ["down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3"]
LATENT_GRIDS = {"g64_main": 8, "g128_main": 4, "g256_main": 2}


def rng(vals: List[float], nd: int = 3) -> str:
    """Range of values as LaTeX 'lo--hi' (single value if equal at nd decimals)."""
    lo, hi = f"{min(vals):.{nd}f}", f"{max(vals):.{nd}f}"
    return lo if lo == hi else f"{lo}--{hi}"


def s2_train() -> None:
    oof = ROOT / "final" / "oof"
    for cond in PRIMARY:
        m = json.loads((oof / f"oof_{cond}.json").read_text())
        logger.info("S2 %-32s ridge train %.3f  (kernel train %.4f; %d folds)", cond,
                    np.mean(m["ridge"]["train_r2"]), np.mean(m["kernel_rbf"]["train_r2"]), len(m["ridge"]["train_r2"]))
    files = sorted(glob.glob(str(oof / "oof_sd15_mid_t500_s*_g128_main.json")))
    assert len(files) == 10, f"expected 10 SD seed files, found {len(files)}"
    per_seed = [np.mean(json.loads(Path(f).read_text())["ridge"]["train_r2"]) for f in files]
    logger.info("S2 %-32s ridge train %.3f  (mean over 10 seeds; range %s)", "sd15_mid_t500_g128_main",
                np.mean(per_seed), rng(per_seed))


def s4_blocks() -> None:
    d = {}
    for f in sorted(glob.glob(str(ROOT / "confirm" / "audit_blocks_*.json"))):
        d |= json.loads(Path(f).read_text())
    assert len(d) == 54, f"expected 54 block-sweep conditions, found {len(d)}"
    for st in STAGES:
        cells = []
        for t in (100, 300, 500):
            vals = [d[f"sd15_{st}_t{t}_s{s}_g128_main"]["random"]["summary"]["ridge"] for s in (0, 1)]
            cells.append(rng(vals))
        logger.info("S4 %-5s & %s \\\\", st, " & ".join(cells))
    perm = [v["random"]["summary"][p] for v in d.values() for p in ("ridge_permuted", "kernel_rbf_permuted")]
    logger.info("S4 permutation-null R²: max %.4f, min %.4f", max(perm), min(perm))


def s6_latent() -> None:
    for t in (100, 300, 500):
        cells = []
        for g in LATENT_GRIDS:
            recs = json.loads((ROOT / "confirm" / f"latent_probe_{g}.json").read_text())
            vals = [r["ridge"] for r in recs if r["rep"] == "pooled" and r["t"] == t and r["seed"] in (0, 1)]
            assert len(vals) == 2, f"{g} t={t}: expected seeds 0 and 1, found {len(vals)}"
            cells.append(rng(vals, nd=2))
        logger.info("S6 t=%d & %s \\\\", t, " & ".join(cells))


def s2_audit() -> None:
    for f in sorted(glob.glob(str(ROOT / "confirm" / "audit_v3_*.json"))):
        for cond, v in json.loads(Path(f).read_text()).items():
            summ = v.get("random", {}).get("summary", v)
            logger.info("S2-audit %s %s %s", Path(f).name, cond,
                        {k: round(x, 4) for k, x in summ.items() if isinstance(x, (int, float))})


def s5_reproduction() -> None:
    path = ROOT / "reproduction_baseline.json"
    d = json.loads(path.read_text())
    logger.info("S5 %s: top-level type %s; preview:\n%s", path, type(d).__name__, json.dumps(d, indent=1)[:3000])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s2_train()
    s4_blocks()
    s6_latent()
    s2_audit()
    s5_reproduction()


if __name__ == "__main__":
    main()
