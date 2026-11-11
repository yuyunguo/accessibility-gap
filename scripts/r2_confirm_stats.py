"""Confirmatory analysis (confirmatory/PROTOCOL.md §7): set-level non-superiority bounds.

Inputs: out-of-fold prediction files from final_probe.py for the confirmatory sets c{m}:
  oof_{model}_nearest_g{G}_c{m}.npz            (224-px models, ImageGPT)
  oof_sd15_mid_t500_s{n}_nearest_g128_c{m}.npz (SD; noise seeds n share set m's stimuli)
Per set m and condition c: g_cm = mean over outer repeats of R²_kernel − R²_ridge (SD: also
averaged over noise seeds); ridge and kernel R² are reported alongside. Per condition: mean, SD, and n over sets; simultaneous one-sided
95% bound UB_c = mean + t_{1−0.05/K, n−1} · SD / √n (Bonferroni over the K conditions of a
family). A family passes at Δ if UB_c < Δ for every condition in it.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import t as student_t

from final_stats import load, point_r2

logger = logging.getLogger(__name__)

DELTA, SENSITIVITY, ALPHA_FW = 0.01, (0.005, 0.02), 0.05
M_224, M_SD, SD_SEEDS = 10, 5, (0, 1, 2, 3, 4)
MODELS_224 = ("clip", "mae", "dino", "supvit_in1k", "supvit_21k", "resnet50")
FAMILIES: Dict[str, List[str]] = {
    "P": [f"{m}_nearest_g56" for m in MODELS_224] + ["imagegpt_nearest_g32", "sd15_mid_t500_g128"],
    "H": [f"{m}_nearest_g28" for m in MODELS_224],
}
FILE_RE = re.compile(r"^oof_(?P<cond>.+?)_c(?P<set>\d+)$")
SD_RE = re.compile(r"^sd15_mid_t(?P<t>\d+)_s(?P<seed>\d+)_nearest_g(?P<g>\d+)$")


def set_r2(path: Path) -> np.ndarray:
    """(ridge R², kernel R², gap) of one OOF file, each averaged over outer repeats."""
    y, oof = load(path)
    r, k = point_r2(y, oof["ridge"]), point_r2(y, oof["kernel_rbf"])
    return np.array([r, k, k - r])


def collect(oof_dir: Path) -> Dict[str, Dict[int, np.ndarray]]:
    """condition -> {set index: (ridge, kernel, gap)}. SD seeds are averaged within a set (all required)."""
    vals: Dict[str, Dict[int, np.ndarray]] = {}
    sd: Dict[tuple, Dict[int, np.ndarray]] = {}
    for f in sorted(oof_dir.glob("oof_*.npz")):
        m = FILE_RE.match(f.stem)
        if not m:
            continue
        cond, s = m.group("cond"), int(m.group("set"))
        sm = SD_RE.match(cond)
        if sm:
            sd.setdefault((f"sd15_mid_t{sm['t']}_g{sm['g']}", s), {})[int(sm["seed"])] = set_r2(f)
        else:
            vals.setdefault(cond, {})[s] = set_r2(f)
    for (cond, s), per_seed in sd.items():
        if tuple(sorted(per_seed)) != SD_SEEDS:
            raise ValueError(f"{cond} set {s}: noise seeds {sorted(per_seed)} != {SD_SEEDS}")
        vals.setdefault(cond, {})[s] = np.mean(list(per_seed.values()), axis=0)
    return vals


def condition_bound(g: np.ndarray, k: int) -> Dict:
    n, mean, sd = len(g), float(g.mean()), float(g.std(ddof=1))
    se = sd / np.sqrt(n)
    return {"n_sets": n, "mean_gap": mean, "sd_gap": sd,
            "ub_simultaneous": mean + float(student_t.ppf(1 - ALPHA_FW / k, n - 1)) * se,
            "ub_marginal": mean + float(student_t.ppf(1 - ALPHA_FW, n - 1)) * se}


def analyse(vals: Dict[str, Dict[int, np.ndarray]]) -> Dict:
    families: Dict[str, Any] = {}
    out = {"delta": DELTA, "sensitivity": list(SENSITIVITY), "families": families}
    for fam, conds in FAMILIES.items():
        missing = [c for c in conds if c not in vals]
        if missing:
            raise ValueError(f"family {fam}: no files for {missing}")
        res: Dict[str, Dict[str, Any]] = {}
        for c in conds:
            want = M_SD if c.startswith("sd15") else M_224
            if len(vals[c]) != want:
                raise ValueError(f"{c}: {len(vals[c])} sets, protocol requires {want}")
            V = np.stack([vals[c][s] for s in sorted(vals[c])])          # sets × (ridge, kernel, gap)
            res[c] = condition_bound(V[:, 2], len(conds))
            res[c]["mean_ridge_r2"], res[c]["mean_kernel_r2"] = float(V[:, 0].mean()), float(V[:, 1].mean())
            res[c]["per_set"] = {str(s): {"ridge": float(v[0]), "kernel": float(v[1]), "gap": float(v[2])}
                                 for s, v in zip(sorted(vals[c]), V)}
        max_ub = max(r["ub_simultaneous"] for r in res.values())
        families[fam] = {"K": len(conds), "conditions": res, "max_ub_simultaneous": max_ub,
                                "passes": {str(d): bool(max_ub < d) for d in (DELTA, *SENSITIVITY)}}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    out = analyse(collect(args.oof_dir))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for fam, r in out["families"].items():
        logger.info("family %s (K=%d): max simultaneous UB %+.4f; passes at Δ=%g: %s", fam, r["K"],
                    r["max_ub_simultaneous"], DELTA, r["passes"][str(DELTA)])


if __name__ == "__main__":
    main()
