"""Reproduction baseline (revision plan §9 A5).

Re-runs the *legacy* probing protocol (fixed 80/20 split, random_state=42,
Ridge / PCA-64 MLP / full-dim MLP) on cached features and checks that every
probe R² matches the value printed in the submitted manuscript
(NEUNET-D-26-04191: Table 1, Table 2, Supplementary Table S1).

SD UNet is excluded: its cached features come from an unseeded diffusion
forward pass and cannot be reproduced by construction (plan §9 A1).
"""

import argparse
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "legacy"))
from revision_probing import probe_all  # noqa: E402  (legacy module)

logger = logging.getLogger(__name__)

# Values as printed in the submitted manuscript (3 decimals).
# Keys: linear_full, linear_pca64, mlp_pca64, mlp_full.
SUBMITTED: Dict[str, Dict[str, float]] = {
    "clip":           {"linear_full": 0.966, "linear_pca64": 0.972, "mlp_pca64": 0.967, "mlp_full": 0.976},
    "mae":            {"linear_full": 0.839, "linear_pca64": 0.921, "mlp_pca64": 0.877, "mlp_full": 0.919},
    "resnet":         {"linear_full": 0.940, "linear_pca64": 0.955, "mlp_pca64": 0.954, "mlp_full": 0.964},
    "autoregressive": {"linear_full": 0.901, "linear_pca64": 0.906, "mlp_pca64": 0.875, "mlp_full": 0.924},
    "dino":           {"linear_full": 0.976, "mlp_full": 0.981},
    "supervised_vit": {"linear_full": 0.932, "mlp_full": 0.968},
}
TOLERANCE = 0.0015  # allows for rounding at the third decimal


def run(cache_dir: Path, out_path: Path, n_bootstrap: int) -> bool:
    """Probe each cached model and compare against submitted values.

    Args:
        cache_dir: Directory containing ``{model}_features.npz`` files.
        out_path: JSON file to write the comparison to.
        n_bootstrap: Bootstrap resamples passed to the legacy prober.

    Returns:
        True if every submitted value is reproduced within tolerance.
    """
    kwargs = {}
    if "n_bootstrap" in inspect.signature(probe_all).parameters:
        kwargs["n_bootstrap"] = n_bootstrap

    report: Dict[str, Dict] = {}
    all_ok = True
    for model, expected in SUBMITTED.items():
        data = np.load(cache_dir / f"{model}_features.npz")
        res = probe_all(data["features"], data["labels"], **kwargs)
        rows = {}
        for key, want in expected.items():
            got = float(res[key]["r2"])
            ok = abs(got - want) <= TOLERANCE
            all_ok &= ok
            rows[key] = {"submitted": want, "reproduced": round(got, 4), "match": ok}
            logger.info(f"{model:15s} {key:13s} submitted={want:.3f} reproduced={got:.4f} {'OK' if ok else 'MISMATCH'}")
        report[model] = rows

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"tolerance": TOLERANCE, "all_match": all_ok, "models": report}, indent=2))
    logger.info(f"All submitted values reproduced: {all_ok}  ->  {out_path}")
    return all_ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "reproduction_baseline.json")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sys.exit(0 if run(args.cache_dir, args.out, args.n_bootstrap) else 1)


if __name__ == "__main__":
    main()
