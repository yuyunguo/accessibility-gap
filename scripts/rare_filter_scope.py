"""Which feature files can the Amendment 1 rare-feature filter change? (read-only)

The filter (final_probe.RareFeatureFilter, --drop-rare f) drops features that are non-zero in fewer
than a fraction f of the rows it is fitted on. It is fitted inside every CV fit, and the smallest
fitting set is the inner-CV fit: n_fit = n_tr - ceil(n_tr / 3) with n_tr = N - ceil(N / 5).
A feature that is non-zero in c of all N rows is non-zero in at least c - (N - n_fit) rows of any
fitting set. If that bound is >= f * n_fit for every feature, the filter keeps every feature in
every fit, the pipeline is numerically unchanged, and the file needs no refit.

Prints one line per file the filter CAN act on ("ACTS") and writes their paths to --list.
Run from the repository root: python scripts/rare_filter_scope.py FILES... --list out.txt
"""

import argparse
import logging
import math
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def can_act(features: np.ndarray, min_frac: float) -> bool:
    """True unless the worst-case bound guarantees the filter keeps every feature in every CV fit."""
    n = features.shape[0]
    n_tr = n - math.ceil(n / 5)
    n_fit = n_tr - math.ceil(n_tr / 3)
    worst = int((features != 0).sum(axis=0).min()) - (n - n_fit)
    return worst < min_frac * n_fit


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("files", type=Path, nargs="+")
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--list", type=Path, required=True, help="output: paths of the files the filter can act on")
    args = ap.parse_args()
    acts: List[str] = []
    for f in args.files:
        d = np.load(f)
        if "features" not in d:
            logger.info("skip (no 'features' array): %s", f)
            continue
        if can_act(d["features"], args.min_frac):
            acts.append(str(f))
            logger.info("ACTS   %s", f)
    args.list.write_text("".join(p + "\n" for p in acts))
    logger.info("%d of %d files can be changed by the filter; list written to %s", len(acts), len(args.files), args.list)


if __name__ == "__main__":
    main()
