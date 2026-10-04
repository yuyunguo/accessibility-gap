"""Invariants the revision analysis depends on. Run: pytest -q tests/"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "stimuli")]

import audit_probes as ap  # noqa: E402
from extract_features_v2 import to_rgb  # noqa: E402
from extract_sd_seeded import ALPHAS, _stim_path  # noqa: E402
from generate_btc_octave import realized_stats  # noqa: E402
from src.probes import ProbeConfig, build_probes, evaluate_folds, stratified_splits  # noqa: E402

N_PER = 200
LABELS = np.repeat(ALPHAS, N_PER).astype(float)


def test_family_index_is_position_within_level():
    fam = ap.family_index(LABELS)
    assert fam.min() == 0 and fam.max() == N_PER - 1
    for a in ALPHAS:
        assert np.array_equal(fam[LABELS == a], np.arange(N_PER))


def test_family_index_rejects_non_contiguous_labels():
    with pytest.raises(AssertionError):
        ap.family_index(np.array([0.1, 0.2, 0.1]))


def test_grouped_outer_splits_never_share_a_family():
    fam = ap.family_index(LABELS)
    for tr, te in ap.outer_splits(LABELS, fam, "grouped", 5, 0):
        assert not set(fam[tr]) & set(fam[te])
        assert set(LABELS[te]) == set(ALPHAS)  # every level present in every test fold


def test_random_outer_splits_are_stratified():
    fam = ap.family_index(LABELS)
    for tr, te in ap.outer_splits(LABELS, fam, "random", 5, 0):
        counts = [np.sum(LABELS[te] == a) for a in ALPHAS]
        assert max(counts) - min(counts) <= 1


def test_inner_cv_folds_contain_every_level():
    """Regression test for the unshuffled inner CV bug (fixed in 787d85e)."""
    from sklearn.model_selection import StratifiedKFold
    fam = ap.family_index(LABELS)
    tr, _ = ap.outer_splits(LABELS, fam, "random", 5, 0)[0]
    lev = np.unique(LABELS[tr], return_inverse=True)[1]
    for itr, iva in StratifiedKFold(3, shuffle=True, random_state=0).split(tr, lev):
        assert set(LABELS[tr][iva]) == set(ALPHAS)
        assert set(LABELS[tr][itr]) == set(ALPHAS)


def test_stratified_splits_cover_all_levels():
    for tr, te in stratified_splits(LABELS, n_splits=5, n_repeats=2, seed=0):
        assert set(LABELS[te]) == set(ALPHAS) and len(set(tr) & set(te)) == 0


def test_exact_replication_and_guard():
    x = (torch.rand(3, 56, 56) > 0.5).float()
    y = to_rgb(x, 224, "nearest", exact=True)
    assert torch.equal(y[:, 0], x.repeat_interleave(4, 1).repeat_interleave(4, 2))
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}
    with pytest.raises(ValueError):
        to_rgb(torch.rand(2, 64, 64), 224, "nearest", exact=True)


def test_v2_stimulus_filenames_match_generator():
    for a in ALPHAS:
        name = _stim_path(Path("d"), a, 56, "main").name
        assert name == f"alpha_{str(float(a)).replace('.', 'p')}_56x56_main.mat"
    assert _stim_path(Path("d"), 0.05, 64, "legacy").name == "informative_alpha_005_64x64.mat"


def test_realized_stats_on_constructed_textures():
    rng = np.random.default_rng(0)
    iid = (rng.random((64, 64, 50)) > 0.5).astype(np.uint8)
    s = realized_stats(iid)
    assert all(abs(v) < 0.02 for v in s.values())
    ones = np.ones((8, 8, 3), dtype=np.uint8)
    assert all(abs(v - 1.0) < 1e-12 for v in realized_stats(ones).values())


def test_probes_recover_linear_signal_and_null_on_permuted_labels():
    rng = np.random.default_rng(0)
    y = np.repeat(np.linspace(0.05, 0.5, 6), 60)
    X = np.outer(y, rng.normal(size=50)) + 0.05 * rng.normal(size=(len(y), 50))
    splits = list(stratified_splits(y, n_splits=5, n_repeats=1, seed=0))
    probes = build_probes(ProbeConfig(n_jobs=1), include_mlp=False)
    r = evaluate_folds(X, y, splits, probes)
    assert np.mean(r["ridge_tuned"]["test_r2"]) > 0.95
    r_null = evaluate_folds(X, rng.permutation(y), splits, probes)
    assert np.mean(r_null["ridge_tuned"]["test_r2"]) < 0.05


def test_realized_stats_geometry_on_checkerboard():
    """Checkerboard s(r,c)=(-1)^(r+c): β_h=β_v=-1, β_d=β_a=+1, θ=-1... α=+1, γ≈0."""
    r, c = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    board = (((r + c) % 2) == 0).astype(np.uint8)
    st = realized_stats(board)
    assert st["beta_h"] == -1 and st["beta_v"] == -1 and st["beta_d"] == 1 and st["beta_a"] == 1
    assert st["alpha"] == 1 and abs(st["gamma"]) < 1e-12


def test_generator_validation_flags_bad_alpha():
    from generate_btc_octave import validate
    good = {k: 0.0 for k in realized_stats(np.ones((4, 4, 2), dtype=np.uint8))} | {"alpha": 0.3}
    assert validate("alpha", 0.3, good, 64, 200)[0]
    assert not validate("alpha", 0.3, good | {"alpha": 0.25}, 64, 200)[0]
    assert not validate("alpha", 0.3, good | {"beta_h": 0.05}, 64, 200)[0]
