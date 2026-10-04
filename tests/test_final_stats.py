"""Correctness tests for the final statistics. Run: pytest -q tests/"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

from final_probe import strata as probe_strata  # noqa: E402
from final_stats import holm, r2_weighted, strata, stratified_weights  # noqa: E402


def test_r2_weighted_with_unit_weights_equals_ordinary_r2():
    rng = np.random.default_rng(0)
    y, pred = rng.normal(size=300), rng.normal(size=300)
    ordinary = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    assert np.isclose(r2_weighted(y, pred, np.ones((1, 300)))[0], ordinary)


def test_r2_weighted_matches_explicit_resample():
    rng = np.random.default_rng(1)
    y, pred = rng.normal(size=50), rng.normal(size=50)
    idx = rng.integers(0, 50, 50)
    w = np.bincount(idx, minlength=50).astype(float)[None, :]
    yb, pb = y[idx], pred[idx]
    explicit = 1 - np.sum((yb - pb) ** 2) / np.sum((yb - yb.mean()) ** 2)
    assert np.isclose(r2_weighted(y, pred, w)[0], explicit)


def test_stratified_weights_preserve_level_counts():
    y = np.repeat([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 200)
    W = stratified_weights(y, np.random.default_rng(0), 100)
    for a in np.unique(y):
        assert np.all(W[:, y == a].sum(1) == 200)


def test_continuous_targets_use_deciles_in_both_scripts():
    y = np.random.default_rng(0).normal(size=5640)
    for f in (strata, probe_strata):
        lev = f(y)
        assert len(np.unique(lev)) == 10
        assert np.all(np.bincount(lev) >= 560)
    W = stratified_weights(y, np.random.default_rng(0), 50)
    assert W.std(1).max() > 0          # bootstrap actually resamples (not the identity)


def test_holm_matches_reference():
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(holm([0.5, 0.5]), [1.0, 1.0])


def test_first_cv_repeat_is_identical_for_mlp_pairing():
    """MLP (1 repeat) is paired with ridge's first repeat; the splits must coincide."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    y = np.repeat(np.arange(6), 200)
    one = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=0).split(y, y))
    ten = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0).split(y, y))[:5]
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(one, ten))
