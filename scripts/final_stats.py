"""Final analysis, step 2: statistics from saved out-of-fold predictions.

Metric: OOF R² per repeat, averaged over repeats. Gap = R²(kernel_rbf) − R²(ridge).
Uncertainty: paired bootstrap over stimuli, stratified by α level (B = 10,000; the
same resample is applied to both probes, every repeat, and every seed). For SD,
seeds are resampled as well (hierarchical: seeds, then stimuli). 95% percentile
CIs; two-sided bootstrap p-values (floor 1/(B+1)) with Holm correction over the
primary family. No effect-size threshold is used.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

B, BOOT_SEED = 10_000, 12345


def r2_weighted(y: np.ndarray, pred: np.ndarray, W: np.ndarray) -> np.ndarray:
    """R² of ``pred`` for each row of bootstrap weights W (B × N); returns length-B array."""
    n = W.sum(1)
    ybar = (W @ y) / n
    ss_tot = W @ (y ** 2) - n * ybar ** 2
    ss_res = W @ ((y - pred) ** 2)
    return 1.0 - ss_res / ss_tot


def strata(y: np.ndarray, max_levels: int = 20) -> np.ndarray:
    """Same stratification as final_probe.py: α level, or deciles for continuous targets."""
    u, inv = np.unique(y, return_inverse=True)
    if len(u) <= max_levels:
        return inv
    edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    return np.searchsorted(edges, y, side="right")


def stratified_weights(y: np.ndarray, rng: np.random.Generator, b: int) -> np.ndarray:
    """Multinomial resampling counts (b × N), resampling within each stratum."""
    W = np.zeros((b, len(y)), dtype=np.float64)
    lev = strata(y)
    for a in np.unique(lev):
        idx = np.flatnonzero(lev == a)
        draws = rng.integers(0, len(idx), size=(b, len(idx)))
        for j in range(len(idx)):
            np.add.at(W, (np.arange(b), idx[draws[:, j]]), 1.0)
    return W


def point_r2(y: np.ndarray, oof: np.ndarray) -> float:
    return float(np.mean([1 - np.sum((y - o) ** 2) / np.sum((y - y.mean()) ** 2) for o in oof]))


def boot_r2(y: np.ndarray, oof: np.ndarray, W: np.ndarray) -> np.ndarray:
    return np.mean([r2_weighted(y, o, W) for o in oof], axis=0)


def summarize(boot_gap: np.ndarray, point_gap: float) -> Dict[str, float]:
    lo, hi = np.percentile(boot_gap, [2.5, 97.5])
    p = 2 * min(np.mean(boot_gap <= 0), np.mean(boot_gap >= 0))
    return {"gap": point_gap, "ci_lo": float(lo), "ci_hi": float(hi), "p_boot": float(max(p, 1 / (B + 1)))}


def holm(ps: List[float]) -> List[float]:
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def load(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = np.load(path)
    return d["y"], {k[4:]: d[k] for k in d.files if k.startswith("oof_")}


def single_condition(path: Path, rng) -> Dict:
    y, oof = load(path)
    W = stratified_weights(y, rng, B)
    res = {p: point_r2(y, o) for p, o in oof.items()}
    bl, bk = boot_r2(y, oof["ridge"], W), boot_r2(y, oof["kernel_rbf"], W)
    res["ridge_ci"] = np.percentile(bl, [2.5, 97.5]).tolist()
    res |= summarize(bk - bl, res["kernel_rbf"] - res["ridge"])
    if "mlp" in oof:
        # MLP uses one CV repeat; it is paired with ridge on that same repeat (identical split).
        r0 = oof["ridge"][:1]
        bm, bl0 = boot_r2(y, oof["mlp"], W), boot_r2(y, r0, W)
        res["gap_mlp"] = res["mlp"] - point_r2(y, r0)
        res["gap_mlp_ci"] = np.percentile(bm - bl0, [2.5, 97.5]).tolist()
    return res


def sd_aggregate(paths: List[Path], rng) -> Dict:
    """Hierarchical bootstrap over seeds and stimuli; stimuli are shared across seeds."""
    ys, oofs = zip(*(load(p) for p in paths))
    assert all(np.array_equal(ys[0], y) for y in ys), "seeds must share the same stimuli in the same order"
    y, k = ys[0], len(paths)
    W = stratified_weights(y, rng, B)
    per_seed_l = np.array([point_r2(y, o["ridge"]) for o in oofs])
    per_seed_k = np.array([point_r2(y, o["kernel_rbf"]) for o in oofs])
    bl = np.stack([boot_r2(y, o["ridge"], W) for o in oofs])        # seeds × B
    bk = np.stack([boot_r2(y, o["kernel_rbf"], W) for o in oofs])
    s = rng.integers(0, k, size=(B, k))                               # resampled seeds per draw
    cols = np.arange(B)[:, None]
    gl, gk = bl.T[cols, s].mean(1), bk.T[cols, s].mean(1)
    res = {"n_seeds": k, "ridge": float(per_seed_l.mean()), "kernel_rbf": float(per_seed_k.mean()),
           "ridge_ci": np.percentile(gl, [2.5, 97.5]).tolist(),
           "ridge_seed_range": [float(per_seed_l.min()), float(per_seed_l.max())],
           "gap_seed_range": [float((per_seed_k - per_seed_l).min()), float((per_seed_k - per_seed_l).max())]}
    res |= summarize(gk - gl, res["kernel_rbf"] - res["ridge"])
    # Variance decomposition of per-(seed, repeat) R² (one-way random effects: seed vs CV repeat).
    def r2_rows(o):
        return np.array([1 - np.sum((y - r) ** 2) / np.sum((y - y.mean()) ** 2) for r in o])
    for probe in ("ridge", "kernel_rbf"):
        M = np.stack([r2_rows(o[probe]) for o in oofs])          # seeds × repeats
        n = M.shape[1]
        ms_b, ms_w = n * np.var(M.mean(1), ddof=1), np.mean(np.var(M, axis=1, ddof=1))
        s2 = max(0.0, (ms_b - ms_w) / n)
        res[f"{probe}_sd_seed"], res[f"{probe}_sd_repeat"] = float(np.sqrt(s2)), float(np.sqrt(ms_w))
        res[f"{probe}_seed_share"] = float(s2 / (s2 + ms_w)) if s2 + ms_w > 0 else 0.0
    if all("mlp" in o for o in oofs):
        # MLP: one CV repeat, paired with ridge on that same repeat; hierarchical over seeds and stimuli.
        pm = np.array([point_r2(y, o["mlp"]) - point_r2(y, o["ridge"][:1]) for o in oofs])
        bm = np.stack([boot_r2(y, o["mlp"], W) - boot_r2(y, o["ridge"][:1], W) for o in oofs])
        res["gap_mlp"] = float(pm.mean())
        res["gap_mlp_ci"] = np.percentile(bm.T[cols, s].mean(1), [2.5, 97.5]).tolist()
        res["gap_mlp_seed_range"] = [float(pm.min()), float(pm.max())]
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oof-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(BOOT_SEED)
    files = sorted(args.oof_dir.glob("oof_*.npz"))
    results, sd_groups = {}, {}
    for f in files:
        m = re.match(r"oof_sd15_mid_t(\d+)_s(\d+)_nearest_g(\d+)_(\w+)$", f.stem)
        if m:
            sd_groups.setdefault((int(m.group(1)), int(m.group(3)), m.group(4)), []).append(f)
        else:
            results[f.stem[4:]] = single_condition(f, rng)
    for (t, g, st), fs in sorted(sd_groups.items()):
        results[f"sd15_mid_t{t}_g{g}_{st}"] = sd_aggregate(sorted(fs), rng)
    primary = [k for k in results if re.search(r"_g56_main$", k) and not k.startswith("sd15")]
    primary += [k for k in results if k.startswith("imagegpt") and k.endswith("_g32_main")]
    primary += [k for k in results if k == "sd15_mid_t500_g128_main"]
    adj = holm([results[k]["p_boot"] for k in primary])
    for k, a in zip(primary, adj):
        results[k]["p_holm"] = a
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"B": B, "boot_seed": BOOT_SEED, "primary_family": primary,
                                    "conditions": results}, indent=1))
    for k in sorted(results):
        r = results[k]
        print(f"{k:42s} ridge={r['ridge']:.3f} kernel={r['kernel_rbf']:.3f} gap={r['gap']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]" + (f" p_holm={r['p_holm']:.4g}" if "p_holm" in r else ""))


if __name__ == "__main__":
    main()
