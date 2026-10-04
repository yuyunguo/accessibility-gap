"""Generate BTC maximum-entropy texture stimuli with GNU Octave and explicit seeds.

Each (statistic, level, set) file is produced in its own Octave call with a
distinct, recorded ``rand('twister', seed)``. The legacy generator never seeded
the RNG, so every file started from the same default stream and stimulus k was
the same texture family at every α level.

Output: ``{out_dir}/{stat}_{level}_{size}x{size}_{set}.mat`` (variable ``maps``,
size × size × nmaps, values in {0, 1}) plus ``manifest.json`` with seeds,
parameters, and validated realized statistics.
"""

import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import scipy.io as sio

logger = logging.getLogger(__name__)

# BTC paramvals order: [g, b, c, d, e, t, u, v, w, a]
STAT_INDEX = {"gamma": 0, "beta1": 1, "beta2": 2, "beta3": 3, "beta4": 4,
              "theta1": 5, "theta2": 6, "theta3": 7, "theta4": 8, "alpha": 9}


def realized_stats(maps: np.ndarray) -> Dict[str, float]:
    """Realized γ, β (h, v, d, a), θ (four orientations), and α in ±1 coding."""
    s = 2.0 * np.transpose(maps, (2, 0, 1)).astype(np.float64) - 1.0
    a, b, c, d = s[:, :-1, :-1], s[:, :-1, 1:], s[:, 1:, :-1], s[:, 1:, 1:]
    return {"gamma": s.mean(), "beta_h": (a * b).mean(), "beta_v": (a * c).mean(),
            "beta_d": (a * d).mean(), "beta_a": (b * c).mean(),
            "theta_abc": (a * b * c).mean(), "theta_abd": (a * b * d).mean(),
            "theta_acd": (a * c * d).mean(), "theta_bcd": (b * c * d).mean(),
            "alpha": (a * b * c * d).mean()}


def run_octave(btc_dir: Path, stat: str, level: float, size: int, nmaps: int, seed: int, out: Path) -> None:
    pv = [0.0] * 10
    pv[STAT_INDEX[stat]] = level
    code = (
        f"addpath('{btc_dir}/EY9314/hoc'); addpath('{btc_dir}/EY7977/STIM'); addpath('{btc_dir}/MATH');"
        f"rand('twister', {seed}); opts=struct(); opts.nrows={size}; opts.ncols={size};"
        f"opts.nmaps={nmaps}; opts.method='btc'; opts.paramvals=[{','.join(map(str, pv))}];"
        f"maps=uint8(hocmake_btc(opts)); save('-mat7-binary', '{out}', 'maps');"
    )
    subprocess.run([os.environ.get("OCTAVE_BIN", "octave"), "--no-gui", "--quiet", "--eval", code],
                   check=True, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--btc-dir", type=Path, default=Path(os.environ.get("BTC_CODE_DIR", "jv")))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stat", default="alpha", choices=sorted(STAT_INDEX))
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--nmaps", type=int, default=200)
    parser.add_argument("--set-name", default="main")
    parser.add_argument("--base-seed", type=int, default=1000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest: List[Dict] = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    used = {m["seed"] for m in manifest}
    seed = max(used | {args.base_seed - 1}) + 1

    for size in args.sizes:
        for level in args.levels:
            name = f"{args.stat}_{str(level).replace('.', 'p')}_{size}x{size}_{args.set_name}.mat"
            out = (args.out_dir / name).resolve()
            run_octave(args.btc_dir.resolve(), args.stat, level, size, args.nmaps, seed, out)
            st = realized_stats(sio.loadmat(out)["maps"])
            manifest.append({"file": name, "stat": args.stat, "level": level, "size": size,
                             "nmaps": args.nmaps, "set": args.set_name, "seed": seed,
                             "realized": {k: round(float(v), 4) for k, v in st.items()}})
            logger.info(f"{name}: seed={seed} realized {args.stat}≈"
                        f"{st['alpha'] if args.stat == 'alpha' else st}")
            manifest_path.write_text(json.dumps(manifest, indent=1))
            seed += 1


if __name__ == "__main__":
    main()
