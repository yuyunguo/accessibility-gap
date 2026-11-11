"""Round-2 Phase 1, item 4: add the imagewise empirical α as an alternative probe target.

Copies each cached feature file and adds ``alpha_check`` (per-image mean four-check
product on the texture grid, ±1 coding, non-wrapping), computed from the stimulus files
in the same order as ``load_alpha_stimuli``. The copy keeps the original file name, so
``final_probe.py --target-key alpha_check`` and ``final_stats.py`` run unchanged.
"""

import argparse
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from r2_stimulus_stats import four_point, load_maps

logger = logging.getLogger(__name__)

SD_GRIDS = (64, 128, 256)


def stimulus_location(feature_file: Path, subdir: Optional[str] = None) -> tuple:
    """(subdir, set name, grid) of the stimuli behind a feature file.

    v3/v3sd files (sets main/rep) map to the stimuli_v2 subdirectories; for other sets
    (confirmatory c{m}, dry) the stimulus subdirectory must be given explicitly.
    """
    m = re.search(r"_g(\d+)_([a-z]+\d*)$", feature_file.stem)
    if not m:
        raise ValueError(f"cannot parse grid/set from {feature_file.name}")
    grid, set_name = int(m.group(1)), m.group(2)
    if subdir is not None:
        return subdir, set_name, grid
    if set_name not in ("main", "rep"):
        raise ValueError(f"{feature_file.name}: set '{set_name}' needs --stim-subdir")
    subdir = "replication" if set_name == "rep" else ("main_sd" if grid in SD_GRIDS and "sd15" in feature_file.stem
                                                       else "main_224")
    return subdir, set_name, grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("features", type=Path, nargs="+")
    parser.add_argument("--stimuli", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--stim-subdir", default=None, help="stimulus subdirectory for sets other than main/rep")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for f in args.features:
        loc = stimulus_location(f, args.stim_subdir)
        if loc not in cache:
            s, y = load_maps(args.stimuli, *loc)
            cache[loc] = (four_point(s), y)
        a_chk, y = cache[loc]
        d = np.load(f)
        if not np.array_equal(d["labels"], y):
            raise ValueError(f"{f.name}: labels differ from the stimulus order {loc}")
        np.savez(args.out_dir / f.name, features=d["features"], labels=d["labels"], alpha_check=a_chk)
        logger.info("%s <- %s: corr(nominal, alpha_check) = %.4f", f.name, loc, np.corrcoef(y, a_chk)[0, 1])


if __name__ == "__main__":
    main()
