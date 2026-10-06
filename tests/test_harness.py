"""Tests for code/harness.py and code/compare.py.  Run: python -m pytest -q tests/test_harness.py"""
import os
import sys

import numpy as np
import pytest
from scipy.stats import beta, binomtest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import harness as H  # noqa: E402
import compare as C  # noqa: E402


# ------------------------------------------------------------------ Clopper-Pearson
def test_cp_zero_failures():
    lo, hi = H.clopper_pearson(0, 100000)
    assert lo == 0.0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 100000), rel=1e-10)


def test_cp_all_failures():
    lo, hi = H.clopper_pearson(100000, 100000)
    assert hi == 1.0
    assert lo == pytest.approx(0.025 ** (1 / 100000), rel=1e-10)


def test_cp_two_of_50000_matches_scipy():
    lo, hi = H.clopper_pearson(2, 50000)
    assert lo == beta.ppf(0.025, 2, 49999) and hi == beta.ppf(0.975, 3, 49998)
    ref = binomtest(2, 50000).proportion_ci(confidence_level=0.95, method="exact")
    assert lo == pytest.approx(ref.low, rel=1e-9) and hi == pytest.approx(ref.high, rel=1e-9)
    assert (round(lo, 8), round(hi, 6)) == (4.84e-6, 1.44e-4)   # values quoted in README


# ------------------------------------------------------------------ McNemar
def test_mcnemar_identical_vectors():
    f = np.zeros(1000, bool)
    f[::7] = True
    r = H.compare(f, f.copy())
    assert (r["b"], r["c"], r["N"], r["delta"], r["p_value"], r["ci95_b_share"]) == (0, 0, 1000, 0.0, 1.0, None)


def test_mcnemar_orientation_known_case():
    cand = np.zeros(100, bool)
    ref = np.zeros(100, bool)
    cand[0:10] = True      # candidate-only failures -> b = 10
    cand[20:25] = True     # both fail (not discordant)
    ref[20:25] = True
    ref[30:33] = True      # reference-only failures -> c = 3
    r = H.compare(cand, ref)
    assert (r["b"], r["c"], r["N"]) == (10, 3, 100)
    assert r["delta"] == pytest.approx(0.07)
    # exact two-sided: 2 * P(X >= 10), X ~ Bin(13, 1/2) = 2 * (286 + 78 + 13 + 1) / 8192
    assert r["p_value"] == pytest.approx(756 / 8192, rel=1e-12)
    assert r["ci95_b_share"] == H.clopper_pearson(10, 13)
    s = H.compare(ref, cand)
    assert (s["b"], s["c"]) == (3, 10) and s["delta"] == pytest.approx(-0.07)
    assert s["p_value"] == pytest.approx(r["p_value"], rel=1e-12)


def test_mcnemar_length_mismatch():
    with pytest.raises(ValueError):
        H.compare(np.zeros(5, bool), np.zeros(6, bool))


def test_packbits_roundtrip_and_cli(tmp_path, capsys):
    rng = np.random.default_rng(0)
    cand, ref = rng.random(1001) < 0.1, rng.random(1001) < 0.1
    H.save_failures(tmp_path / "c.npy", cand)
    H.save_failures(tmp_path / "r.npy", ref)
    assert np.array_equal(H.load_failures(tmp_path / "c.npy", 1001), cand)
    with pytest.raises(ValueError):
        H.load_failures(tmp_path / "c.npy", 1000 - 8)
    C.main([str(tmp_path / "c.npy"), str(tmp_path / "r.npy"), "--shots", "1001"])
    out = capsys.readouterr().out
    want = H.compare(cand, ref)
    assert f"b      = {want['b']}" in out and f"c      = {want['c']}" in out and "N      = 1001" in out


# ------------------------------------------------------------------ latency
def test_latency_keys_order_and_warmup_frames():
    frames = np.arange(1100, dtype=np.int64)[:, None] * np.ones((1, 64), dtype=np.int64)
    seen = []

    def decode_one(f):
        seen.append(int(f[0]))
        return np.sort(f[::-1] * 3 % 7)          # deterministic CPU work, no sleep

    r = H.latency(decode_one, frames, warmup=100)
    assert seen == list(range(1100))              # warm-up on frames 0..99, timed on 100..1099
    assert r["n"] == 1000 and r["warmup"] == 100 and r["unit"] == "us"
    for k in ("n", "mean", "p50", "p90", "p95", "p99", "max", "ci95"):
        assert k in r
    assert 0 < r["p50"] <= r["p90"] <= r["p95"] <= r["p99"] <= r["max"]
    for q in ("p50", "p95", "p99"):
        c = r["ci95"][q]
        assert c["coverage"] >= 0.95
        assert c["lo"] is not None and c["hi"] is not None and c["lo"] <= r[q] <= c["hi"]
    with pytest.raises(ValueError):
        H.latency(decode_one, frames[:100], warmup=100)


# ------------------------------------------------------------------ output-kind detection
class _Batch:
    def __init__(self, out):
        self.out = out

    def decode_batch(self, frames):
        return self.out


def test_scoring_oracles_and_one_flipped_label():
    d = H.load_frozen("B", code="bb72", p=0.04)
    assert H.score(_Batch(d.truth), d)["failures"] == 0
    assert H.score(_Batch(d.truth), d)["kind"] == "correction"
    assert H.score(_Batch(d.labels), d)["kind"] == "logical"
    bad = d.labels.copy()
    bad[17, 3] ^= 1
    r = H.score(_Batch(bad), d)
    assert r["failures"] == 1 and r["fail"][17]
    zeros = H.score(_Batch(np.zeros_like(d.labels)), d)
    assert zeros["failures"] == int(d.labels.any(axis=1).sum())
    with pytest.raises(ValueError):
        H.score(_Batch(np.zeros((d.shots, 5), np.uint8)), d)


def test_arm_c_spacetime_reduction():
    d = H.load_frozen("C", p=0.02)
    n, m, T = d.truth.shape[1], d.HZ.shape[0], d.T
    st = np.zeros((d.shots, T * n + T * m), np.uint8)
    st[:, :n] = d.truth                  # whole error placed in round 0
    st[:, n:2 * n] ^= 1                  # same flip in rounds 1 and 2 cancels in the XOR
    st[:, 2 * n:3 * n] ^= 1
    st[:, T * n:] = 1                    # measurement-error columns are ignored
    r = H.score(_Batch(st), d)
    assert r["kind"] == "spacetime_correction" and r["failures"] == 0
    assert H.score(_Batch(d.truth), d)["failures"] == 0


# ------------------------------------------------------------------ reference counts (frozen CSVs)
@pytest.mark.parametrize("kw,expected", [
    (dict(arm="A", d=5, p=0.001), 78),
    (dict(arm="B", code="bb72", p=0.04), 3969),
    (dict(arm="B", code="bb144", p=0.02), 2),
    (dict(arm="C", p=0.02), 106),
])
def test_reference_counts(kw, expected):
    data = H.load_frozen(**kw)
    r = H.score(H.reference_for(data), data)
    assert (r["failures"], r["shots"]) == (expected, data.shots)
    assert r["ci95"] == H.clopper_pearson(expected, data.shots)
