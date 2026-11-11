"""Correctness tests for the round-2 Phase 1 scripts. Run: pytest -q tests/"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "stimuli")]

from generate_btc_octave import realized_stats  # noqa: E402
from r2_lolo import lolo  # noqa: E402
from r2_simultaneous import bounds  # noqa: E402
from r2_stimulus_stats import four_point, nine_stats, replicate  # noqa: E402
from final_probe import probes_for  # noqa: E402


def _random_pm1(n: int, g: int, seed: int) -> np.ndarray:
    return np.where(np.random.default_rng(seed).random((n, g, g)) < 0.5, -1.0, 1.0)


@pytest.mark.parametrize("g,k", [(28, 8), (56, 4), (112, 2), (64, 8), (256, 2), (32, 1)])
def test_replication_identity(g, k):
    """Reviewer's identity: input-scale α = 1 − ((G−1)/(kG−1))² (1 − check-scale α)."""
    s = _random_pm1(4, g, g)
    expected = 1 - ((g - 1) / (k * g - 1)) ** 2 * (1 - four_point(s))
    assert np.allclose(four_point(replicate(s, k)), expected, atol=1e-12)


def test_four_point_matches_generator_definition():
    s = _random_pm1(3, 40, 1)
    maps = np.transpose((s + 1) / 2, (1, 2, 0))          # generator layout (rows, cols, nmaps), {0, 1}
    assert np.isclose(four_point(s).mean(), realized_stats(maps)["alpha"])


def test_wrap_variant_counts_every_block_once():
    s = _random_pm1(2, 16, 2)
    manual = np.zeros(2)
    for i in range(16):
        for j in range(16):
            manual += s[:, i, j] * s[:, (i + 1) % 16, j] * s[:, i, (j + 1) % 16] * s[:, (i + 1) % 16, (j + 1) % 16]
    assert np.allclose(four_point(s, wrap=True), manual / 256)


def test_replicate_equals_pipeline_nearest_resize():
    torch = pytest.importorskip("torch")
    from extract_features_v2 import to_rgb
    s = _random_pm1(2, 56, 3)
    x = to_rgb(torch.from_numpy(((s + 1) / 2).astype(np.float32)), 224, "nearest", exact=True)[:, 0].numpy()
    assert np.array_equal(2 * x - 1, replicate(s, 4))


def test_nine_stats_shape_and_known_values():
    s = np.ones((1, 8, 8))
    st = nine_stats(s)
    assert st.shape == (1, 9)
    assert st[0, 0] == 1 and st[0, 1] == 0 and np.all(st[0, 6:] == 1) and np.all(st[0, 4:6] == 0)


def test_simultaneous_bound_reduces_to_marginal_for_one_condition():
    draws = np.random.default_rng(4).normal(0.01, 0.002, 10_000)
    out = bounds({"c": {"gap": 0.01, "draws": draws}}, ["c"])["conditions"]["c"]
    assert np.isclose(out["ub_simultaneous"], out["ub_marginal"])


def test_simultaneous_bound_widens_with_family_size():
    rng = np.random.default_rng(5)
    res = {f"c{i}": {"gap": 0.0, "draws": rng.normal(0, 0.001, 10_000)} for i in range(8)}
    one, eight = bounds(res, ["c0"]), bounds(res, list(res))
    assert eight["q_simultaneous"] > one["q_simultaneous"]


def _write_oof(path: Path, gap: float, seed: int) -> None:
    """A synthetic OOF file whose kernel − ridge R² difference is exactly ``gap``."""
    rng = np.random.default_rng(seed)
    y = np.repeat([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 20)
    sst = np.sum((y - y.mean()) ** 2)
    e = rng.normal(size=len(y))
    e -= e.mean()
    ridge = y + e * np.sqrt(0.1 * sst / np.sum(e ** 2))             # R² = 0.9
    kern = y + e * np.sqrt((0.1 - gap) * sst / np.sum(e ** 2))      # R² = 0.9 + gap
    np.savez(path, y=y, oof_ridge=np.tile(ridge, (2, 1)), oof_kernel_rbf=np.tile(kern, (2, 1)))


def test_confirm_stats_end_to_end(tmp_path):
    from r2_confirm_stats import FAMILIES, M_224, M_SD, SD_SEEDS, analyse, collect
    for cond in FAMILIES["P"] + FAMILIES["H"]:
        if cond.startswith("sd15"):
            for m in range(M_SD):
                for s in SD_SEEDS:
                    _write_oof(tmp_path / f"oof_sd15_mid_t500_s{s}_nearest_g128_c{m}.npz", 0.001 * (m % 2), m)
        else:
            for m in range(M_224):
                _write_oof(tmp_path / f"oof_{cond}_c{m}.npz", 0.002 + 0.001 * (m % 2), m)
    out = analyse(collect(tmp_path))
    c = out["families"]["H"]["conditions"]["mae_nearest_g28"]
    assert np.isclose(c["mean_gap"], 0.0025) and c["n_sets"] == M_224
    assert c["ub_simultaneous"] > c["ub_marginal"] > c["mean_gap"]
    assert out["families"]["P"]["passes"]["0.01"] is True
    assert out["families"]["H"]["passes"]["0.005"] is True               # UB ≈ 0.003 < 0.005
    sd = out["families"]["P"]["conditions"]["sd15_mid_t500_g128"]
    assert sd["n_sets"] == M_SD and np.isclose(sd["mean_gap"], 0.0004)  # seeds averaged within set


def test_confirm_stats_rejects_missing_sets(tmp_path):
    from r2_confirm_stats import FAMILIES, analyse, collect
    for cond in FAMILIES["P"] + FAMILIES["H"]:
        if not cond.startswith("sd15"):
            _write_oof(tmp_path / f"oof_{cond}_c0.npz", 0.0, 0)
    with pytest.raises(ValueError):
        analyse(collect(tmp_path))


def test_default_probes_unchanged_by_amendment():
    for name, (pipe, _) in probes_for(64, True).items():
        assert [s for s, _ in pipe.steps] == ["scale", "est"], name


def test_rare_filter_drops_only_rare_columns():
    from final_probe import RareFeatureFilter
    X = np.zeros((200, 4))
    X[:, 0] = 1.0                       # dense
    X[:1, 1] = 0.01                     # non-zero in 0.5% of rows -> dropped at 1%
    X[:2, 2] = 0.01                     # 1% -> kept
    f = RareFeatureFilter(0.01).fit(X)
    assert f.keep_.tolist() == [True, False, True, False]
    assert f.transform(X).shape == (200, 2)


def test_rare_filter_prevents_ridge_blowup():
    """A unit firing in one training image and in a test image must not dominate the prediction."""
    from sklearn.base import clone
    rng = np.random.default_rng(7)
    y = np.repeat([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 40)
    X = np.column_stack([y + rng.normal(0, 0.02, len(y)), rng.normal(size=(len(y), 5)), np.zeros(len(y))])
    tr, te = np.arange(len(y)) % 5 != 0, np.arange(len(y)) % 5 == 0
    X[np.flatnonzero(tr)[0], -1] = 1e-4          # fires once in training
    X[np.flatnonzero(te)[0], -1] = 5e-3          # fires in a test image, 50x larger
    for drop, ok in ((None, False), (0.01, True)):
        pipe, _ = probes_for(X.shape[1], False, drop)["ridge"]
        p = clone(pipe).set_params(est__alpha=1e-2).fit(X[tr], y[tr]).predict(X[te])
        assert bool(np.abs(p - y[te]).max() < 0.2) == ok      # unfiltered: 0.30; filtered: 0.03


def test_lolo_recovers_a_linear_target():
    rng = np.random.default_rng(6)
    y = np.repeat([0.05, 0.1, 0.2, 0.3, 0.4, 0.5], 40)
    X = np.column_stack([y + rng.normal(0, 0.01, len(y)), rng.normal(size=(len(y), 4))])
    pipe, grid = probes_for(X.shape[1], False)["ridge"]
    res = lolo(X, y, pipe, grid, n_jobs=1)
    assert res["r2_pooled"] > 0.95 and set(res["per_level"]) == {"0.05", "0.1", "0.2", "0.3", "0.4", "0.5"}
