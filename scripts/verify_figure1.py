"""Independent read-back check of Figure 1: every plotted point, CI, label, and fill vs final_stats.json.

Rebuilds both panels with make_figure1, reads the values back from the matplotlib artists, and
compares them with the JSON through a label -> key mapping written separately from
make_figure1.parse. Exits 1 on any mismatch.

Run: python scripts/verify_figure1.py results/final/final_stats.json
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figure1 as mf  # noqa: E402

logger = logging.getLogger(__name__)

NAME = {"CLIP": "clip", "DINO": "dino", "MAE": "mae", "ResNet50": "resnet50",
        "ViT-B/16, 1k": "supvit_in1k", "ViT-B/16, 21k": "supvit_21k", "ImageGPT": "imagegpt"}
G224 = {8: "g28", 4: "g56", 2: "g112"}
GSD = {8: "g64", 4: "g128", 2: "g256"}


def key_from_label(text: str) -> str:
    """Map a panel-B row label (e.g. 'SD mid, t = 500, k = 4, rep.') to its JSON condition key."""
    split = "rep" if text.endswith(", rep.") else "main"
    text = text.replace(", rep.", "")
    m = re.match(r"SD mid, t = (\d+), k = (\d)$", text)
    if m:
        return f"sd15_mid_t{m[1]}_{GSD[int(m[2])]}_{split}"
    base, k = re.match(r"(.*), k = (\d)$", text).groups()
    grid = "g32" if base == "ImageGPT" else G224[int(k)]
    return f"{NAME[base]}_nearest_{grid}_{split}"


def check_panel_b(ax, cond: Dict, primary: set) -> int:
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 35 and len(set(labels)) == 35, "panel B must have 35 distinct rows"
    assert len(ax.containers) == 35, "panel B must have 35 error-bar artists"
    fails, seen = 0, set()
    for c in ax.containers:
        line, _, bars = c.lines
        x, row = line.get_xdata()[0], int(line.get_ydata()[0])
        key = key_from_label(labels[row])
        seen.add(key)
        v = cond[key]
        seg = bars[0].get_segments()[0]
        lo, hi = sorted([seg[0][0], seg[1][0]])
        filled = line.get_markerfacecolor() != "white"
        ok = (np.isclose(x, v["gap"]) and np.isclose(lo, v["ci_lo"]) and np.isclose(hi, v["ci_hi"])
              and filled == (key in primary))
        if not ok:
            fails += 1
            logger.error("B mismatch: %s -> %s", labels[row], key)
    assert len(seen) == 35, "each row must map to a distinct condition"
    n_filled = sum(c.lines[0].get_markerfacecolor() != "white" for c in ax.containers)
    logger.info("panel B: 35 rows checked; filled = %d", n_filled)
    return fails + (n_filled != len(primary))


def check_panel_a(ax, cond: Dict) -> int:
    ticks = [int(t.get_text()) for t in ax.get_xticklabels()]
    fails, n_points = 0, 0
    for c in ax.containers:
        line, _, bars = c.lines
        lab = c.get_label()
        for xi, yi, seg in zip(line.get_xdata(), line.get_ydata(), bars[0].get_segments()):
            assert abs(xi - round(xi)) <= mf.DODGE + 1e-9, "dodge must not move a point past its tick"
            k = ticks[int(round(xi))]
            if lab.startswith("SD"):
                key = f"sd15_mid_t{lab.split('= ')[1]}_{GSD[k]}_main"
            else:
                key = f"{NAME[lab]}_nearest_{'g32' if lab == 'ImageGPT' else G224[k]}_main"
            v = cond[key]
            lo, hi = sorted([seg[0][1], seg[1][1]])
            if not (np.isclose(yi, v["ridge"]) and np.isclose(lo, v["ridge_ci"][0])
                    and np.isclose(hi, v["ridge_ci"][1])):
                fails += 1
                logger.error("A mismatch: %s k=%d", lab, k)
            n_points += 1
    logger.info("panel A: series = %d, points = %d", len(ax.containers), n_points)
    return fails + (n_points != 28)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = Path(sys.argv[1])
    data = json.loads(path.read_text())
    rows, primary = mf.load(path)
    _, (a, b) = plt.subplots(1, 2)
    mf.panel_a(a, rows)
    mf.panel_b(b, rows, primary)
    fails = check_panel_a(a, data["conditions"]) + check_panel_b(b, data["conditions"], set(data["primary_family"]))
    logger.info("FAILS = %d", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
