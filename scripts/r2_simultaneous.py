"""Round-2 Phase 1, item 6: simultaneous one-sided upper bounds for the kernel − ridge gap.

The bootstrap draws of final_stats.py were not saved. This script regenerates them by
replaying final_stats.py's random-number sequence exactly (same seed, same file order,
same calls), and checks that every point gap and percentile CI matches
``final_stats.json`` before using the draws.

For a family F of conditions, with gap_hat_c the point gap and se_c the bootstrap SD:
    T_b = max_{c in F} (gap_hat_c − gap_b,c) / se_c,   q = 95% quantile of T_b,
    simultaneous one-sided 95% upper bound_c = gap_hat_c + q · se_c.
The marginal analogue (q_c from each condition alone) is reported for comparison.
These bounds are conditional on the fitted out-of-fold predictions (no refitting).
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from final_stats import B, BOOT_SEED, boot_r2, load, point_r2, stratified_weights

logger = logging.getLogger(__name__)
SD_RE = re.compile(r"oof_sd15_mid_t(\d+)_s(\d+)_nearest_g(\d+)_(\w+)$")


def single_draws(path: Path, rng) -> Dict:
    y, oof = load(path)
    W = stratified_weights(y, rng, B)
    bl, bk = boot_r2(y, oof["ridge"], W), boot_r2(y, oof["kernel_rbf"], W)
    return {"gap": point_r2(y, oof["kernel_rbf"]) - point_r2(y, oof["ridge"]), "draws": bk - bl}


def sd_draws(paths: List[Path], rng) -> Dict:
    ys, oofs = zip(*(load(p) for p in paths))
    y, k = ys[0], len(paths)
    W = stratified_weights(y, rng, B)
    bl = np.stack([boot_r2(y, o["ridge"], W) for o in oofs])
    bk = np.stack([boot_r2(y, o["kernel_rbf"], W) for o in oofs])
    s = rng.integers(0, k, size=(B, k))
    cols = np.arange(B)[:, None]
    gap = np.mean([point_r2(y, o["kernel_rbf"]) - point_r2(y, o["ridge"]) for o in oofs])
    return {"gap": float(gap), "draws": bk.T[cols, s].mean(1) - bl.T[cols, s].mean(1)}


def replay(oof_dir: Path) -> Dict[str, Dict]:
    """Bootstrap gap draws for every condition, in final_stats.py's exact RNG order."""
    rng = np.random.default_rng(BOOT_SEED)
    res, groups = {}, {}
    for f in sorted(oof_dir.glob("oof_*.npz")):
        m = SD_RE.match(f.stem)
        if m:
            groups.setdefault((int(m.group(1)), int(m.group(3)), m.group(4)), []).append(f)
        else:
            res[f.stem[4:]] = single_draws(f, rng)
    for (t, g, st), fs in sorted(groups.items()):
        res[f"sd15_mid_t{t}_g{g}_{st}"] = sd_draws(sorted(fs), rng)
    return res


def bounds(res: Dict[str, Dict], family: List[str]) -> Dict:
    gap = np.array([res[c]["gap"] for c in family])
    D = np.stack([res[c]["draws"] for c in family])               # family × B
    se = D.std(axis=1, ddof=1)
    Z = (gap[:, None] - D) / se[:, None]
    q_sim = float(np.quantile(Z.max(axis=0), 0.95))
    q_marg = np.quantile(Z, 0.95, axis=1)
    return {"q_simultaneous": q_sim, "family_size": len(family),
            "conditions": {c: {"gap": float(gap[i]), "se": float(se[i]),
                               "ub_marginal": float(gap[i] + q_marg[i] * se[i]),
                               "ub_simultaneous": float(gap[i] + q_sim * se[i])} for i, c in enumerate(family)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True, help="final_stats.json to verify against")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ref = json.loads(args.stats.read_text())
    res = replay(args.oof_dir)
    assert set(res) == set(ref["conditions"]), "condition sets differ from final_stats.json"
    worst = 0.0
    for c, r in res.items():
        lo, hi = np.percentile(r["draws"], [2.5, 97.5])
        rc = ref["conditions"][c]
        worst = max(worst, abs(r["gap"] - rc["gap"]), abs(lo - rc["ci_lo"]), abs(hi - rc["ci_hi"]))
    assert worst < 1e-10, f"replay does not reproduce final_stats.json (max diff {worst:.2e})"
    logger.info("replay reproduces final_stats.json (max diff %.1e)", worst)
    btc = sorted(c for c in res if not c.startswith("nat_"))
    assert len(btc) == 35, f"expected 35 controlled-texture conditions, found {len(btc)}"
    out = {"B": B, "boot_seed": BOOT_SEED, "replay_max_diff": worst,
           "primary": bounds(res, ref["primary_family"]), "all35": bounds(res, btc)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    for fam in ("primary", "all35"):
        ub = [v["ub_simultaneous"] for v in out[fam]["conditions"].values()]
        logger.info("%s: q=%.3f, max simultaneous UB %+.4f", fam, out[fam]["q_simultaneous"], max(ub))


if __name__ == "__main__":
    main()
