"""Confirm whether the accessibility gap survives matched, tuned probes.

For each feature file: legacy probes (Ridge λ=1, MLP L2=0.1) and tuned probes
(RidgeCV; MLP with inner-CV L2), evaluated on repeated stratified 5-fold CV.
Ridge probes run on all folds; MLP probes on the first ``--mlp-folds`` folds
(identical folds, so gaps are paired).
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from src.probes import ProbeConfig, build_probes, evaluate_folds, stratified_splits, summarize  # noqa: E402

logger = logging.getLogger(__name__)


def confirm_file(path: Path, cfg: ProbeConfig, n_repeats: int, mlp_folds: int) -> Dict:
    """Run legacy and tuned probes on one ``.npz`` feature file."""
    data = np.load(path)
    X, y = data["features"].astype(np.float64), data["labels"].astype(np.float64)
    splits = list(stratified_splits(y, n_splits=5, n_repeats=n_repeats, seed=cfg.seed))
    probes = build_probes(cfg, include_mlp=True)
    ridge = {k: v for k, v in probes.items() if k.startswith("ridge")}
    mlps = {k: v for k, v in probes.items() if k.startswith("mlp")}

    logger.info(f"{path.name}: X={X.shape}, ridge on {len(splits)} folds")
    folds = evaluate_folds(X, y, splits, ridge)
    if mlp_folds > 0:
        logger.info(f"{path.name}: MLP on first {mlp_folds} folds")
        folds.update(evaluate_folds(X, y, splits[:mlp_folds], mlps))
    return {"file": path.name, "n": int(len(y)), "dim": int(X.shape[1]),
            "summary": summarize(folds), "folds": folds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", nargs="+", type=Path, help=".npz files with 'features' and 'labels'")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--mlp-folds", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    cfg = ProbeConfig(n_jobs=args.n_jobs)
    results: List[Dict] = []
    for f in args.features:
        res = confirm_file(f, cfg, args.n_repeats, args.mlp_folds)
        results.append(res)
        s = res["summary"]
        logger.info(
            f"RESULT {f.stem}: ridge_l1={s['ridge_fixed_l1']['test_mean']:.3f} "
            f"ridge_tuned={s['ridge_tuned']['test_mean']:.3f} "
            f"mlp_l0.1={s.get('mlp_fixed_l0.1', {}).get('test_mean', float('nan')):.3f} "
            f"mlp_tuned={s.get('mlp_tuned', {}).get('test_mean', float('nan')):.3f} "
            f"gap_legacy={s.get('gap_legacy', {}).get('mean', float('nan')):+.3f} "
            f"gap_tuned={s.get('gap_tuned', {}).get('mean', float('nan')):+.3f}"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"config": cfg.__dict__ | {"ridge_alphas": list(cfg.ridge_alphas)},
                                        "results": results}, indent=1, default=float))


if __name__ == "__main__":
    main()
