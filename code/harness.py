"""Evaluation harness: score any decoder on the frozen validation sets and compare two
decoders shot by shot.

Arms (frozen arrays in data/frozen_validation/, construction as in code/verify_redecode.py):
  A  surface_d{d}_p0.001_r25_s100000   decoder input: Stim detector frame (uint8)
                                       scoring: observable 0 (obs[:, 0])
  B  qldpc_{code}_p{p}_s50000          decoder input: syndrome s = H_Z e (uint8, length m)
                                       scoring: data X error e (length n)
  C  phenom_bb72_T6_p{p}_q{p}_s2000    decoder input: space-time detector bits (length (T+1)*m)
                                       scoring: total data X error E = e_0 ^ ... ^ e_{T-1}

A decoder is any object with decode(frame) -> 1-D array of 0/1 values. The harness reads the
output length to decide what it is:
  Arm A: length 1                -> predicted flip of observable 0. Fail iff != obs[:, 0].
  Arm B: length n                -> physical correction e_hat.
         length k (=12)          -> logical prediction of L_Z e (mod 2).
  Arm C: length n                -> net data correction E_hat.
         length T*n + T*m        -> full space-time correction [e_0..e_{T-1} | f_0..f_{T-1}];
                                    reduced to E_hat = XOR_r e_hat_r (the first T*n entries,
                                    reshaped (T, n), XOR over rounds), as in verify_redecode.py.
         length k (=12)          -> logical prediction of L_Z E (mod 2).
Correction outputs fail iff the residual (truth XOR correction) has odd overlap with any row of
L_Z. Logical outputs fail iff any predicted bit differs from L_Z (truth) mod 2.

L_Z is the fixed basis published in data/logicals/{code}_LZ.npy (exported from
code/prepare_qldpc.css_logicals with ldpc 2.4.1; see export_logicals()). For a correction that
reproduces the syndrome the failure decision does not depend on which L_Z basis is used; for a
correction that does not (for example plain BP that did not converge) or for a logical
prediction, it does, which is why the basis is fixed and published.

An optional decode_batch(frames) -> 2-D array is used when present (score(batch=...)).
"""
from __future__ import annotations

import glob
import hashlib
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.stats import beta, binom, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FROZEN_DIR = os.path.join(ROOT, "data", "frozen_validation")
LOGICALS_DIR = os.path.join(ROOT, "data", "logicals")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Paper settings (identical to verify_redecode.py / baseline_bposd*.py)
BP_KW = dict(max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625, schedule="serial")
SURFACE_P = 0.001
SURFACE_ROUNDS = 25
PHENOM_T = 6


# --------------------------------------------------------------------------- statistics
def clopper_pearson(f: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact two-sided equal-tailed Clopper-Pearson interval for f successes in n trials.
    Lower limit 0 when f == 0, upper limit 1 when f == n (scipy.stats.beta quantiles)."""
    f, n = int(f), int(n)
    if n <= 0 or f < 0 or f > n:
        raise ValueError(f"need 0 <= f <= n and n > 0, got f={f}, n={n}")
    lo = 0.0 if f == 0 else float(beta.ppf(alpha / 2, f, n - f + 1))
    hi = 1.0 if f == n else float(beta.ppf(1 - alpha / 2, f + 1, n - f))
    return lo, hi


def compare(fail_candidate, fail_reference) -> dict:
    """Paired (same-shot) comparison of two decoders' per-shot failure vectors.

    b = shots where the candidate fails and the reference succeeds,
    c = shots where the candidate succeeds and the reference fails,
    delta = (b - c) / N = LER(candidate) - LER(reference) on these N shots.
    p_value: exact two-sided McNemar test, scipy.stats.binomtest(b, b + c, 0.5).pvalue
             (1.0 when b + c == 0).
    ci95_b_share: exact Clopper-Pearson 95% interval for b / (b + c) (None when b + c == 0).

    Marginal Clopper-Pearson intervals from score() are for reporting each decoder's LER.
    Claims that one decoder is better than another on the frozen shots use this paired test.
    """
    a = np.asarray(fail_candidate).astype(bool).ravel()
    r = np.asarray(fail_reference).astype(bool).ravel()
    if a.shape != r.shape:
        raise ValueError(f"failure vectors differ in length: {a.size} vs {r.size}")
    n = int(a.size)
    b = int(np.sum(a & ~r))
    c = int(np.sum(~a & r))
    if b + c == 0:
        p, ci = 1.0, None
    else:
        p, ci = float(binomtest(b, b + c, 0.5).pvalue), clopper_pearson(b, b + c)
    return {"b": b, "c": c, "N": n, "delta": (b - c) / n if n else float("nan"),
            "p_value": p, "ci95_b_share": ci,
            "fail_candidate": int(a.sum()), "fail_reference": int(r.sum())}


def latency(decode_one, frames, warmup: int = 500) -> dict:
    """Online single-call latency in microseconds.

    frames[:warmup] are decoded first and not timed; every remaining frame (distinct from the
    warm-up frames) is decoded in its own call, timed with time.perf_counter.
    Returns n (timed calls), mean, p50, p90, p95, p99 (numpy.percentile, linear interpolation),
    max, and ci95: for p50/p95/p99 a distribution-free interval [x_(l), x_(u+1)] from order
    statistics (1-based ranks), with l = binom.ppf(0.025, n, q) and u = binom.ppf(0.975, n, q);
    its coverage F(u) - F(l - 1) >= 0.95 (F the Binomial(n, q) CDF) is returned as well.
    A limit is None when the required rank falls outside 1..n. The intervals assume the timed
    calls are i.i.d. draws from one stationary distribution; drift, other load on the machine
    or dependence between calls is not accounted for.
    """
    frames = np.asarray(frames)
    if frames.shape[0] <= warmup:
        raise ValueError(f"need more than warmup={warmup} frames, got {frames.shape[0]}")
    for i in range(warmup):
        decode_one(frames[i])
    timed = frames[warmup:]
    lat = np.empty(timed.shape[0], dtype=np.float64)
    for i in range(timed.shape[0]):
        x = timed[i]
        t0 = time.perf_counter()
        decode_one(x)
        lat[i] = (time.perf_counter() - t0) * 1e6
    n = int(lat.size)
    srt = np.sort(lat)
    out = {"n": n, "unit": "us", "warmup": int(warmup), "mean": float(lat.mean())}
    for q in (50, 90, 95, 99):
        out[f"p{q}"] = float(np.percentile(lat, q))
    out["max"] = float(srt[-1])
    ci = {}
    for q in (50, 95, 99):
        qq = q / 100.0
        l = int(binom.ppf(0.025, n, qq))
        u = int(binom.ppf(0.975, n, qq))
        lo = float(srt[l - 1]) if l >= 1 else None
        hi = float(srt[u]) if u + 1 <= n else None
        cov = float(binom.cdf(u, n, qq) - (binom.cdf(l - 1, n, qq) if l >= 1 else 0.0))
        ci[f"p{q}"] = {"lo": lo, "hi": hi, "rank_lo": l, "rank_hi": u + 1, "coverage": cov}
    out["ci95"] = ci
    return out


# --------------------------------------------------------------------------- data
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_sum(path: str, sums_file: str) -> None:
    name = os.path.basename(path)
    with open(sums_file) as fh:
        sums = {ln.split()[1]: ln.split()[0] for ln in fh if ln.strip()}
    if name not in sums:
        raise RuntimeError(f"{name} not listed in {sums_file}")
    got = _sha256(path)
    if got != sums[name]:
        raise RuntimeError(f"sha256 mismatch for {name}: {got} != {sums[name]}")


def _one(pattern: str) -> str:
    hits = sorted(glob.glob(os.path.join(FROZEN_DIR, pattern)))
    if len(hits) != 1:
        raise FileNotFoundError(f"{pattern}: {len(hits)} files in {FROZEN_DIR}")
    return hits[0]


def load_logicals(code: str) -> np.ndarray:
    """Published L_Z basis (uint8, k x n) for 'bb72' or 'bb144'; sha256 is checked against
    data/logicals/SHA256SUMS and the rows are checked to lie in ker(H_X) and to be independent
    of rowspace(H_Z) (k = 12)."""
    from ldpc import mod2
    import prepare_qldpc as Q
    path = os.path.join(LOGICALS_DIR, f"{code}_LZ.npy")
    _check_sum(path, os.path.join(LOGICALS_DIR, "SHA256SUMS"))
    LZ = np.load(path)
    HX, HZ = Q.build_bb_code(code)
    assert LZ.dtype == np.uint8 and LZ.shape == (12, HX.shape[1])
    assert not ((HX.astype(int) @ LZ.T.astype(int)) % 2).any()
    rz = mod2.rank(sp.csr_matrix(HZ))
    assert mod2.rank(sp.csr_matrix(np.vstack([HZ, LZ]).astype(np.uint8))) == rz + 12
    return LZ


def export_logicals() -> None:
    """Write data/logicals/{bb72,bb144}_LZ.npy from prepare_qldpc.css_logicals (installed ldpc)
    and data/logicals/SHA256SUMS. Run once: python code/harness.py --export-logicals"""
    import prepare_qldpc as Q
    os.makedirs(LOGICALS_DIR, exist_ok=True)
    lines = []
    for code in ("bb72", "bb144"):
        HX, HZ = Q.build_bb_code(code)
        _, LZ = Q.css_logicals(HX, HZ)
        path = os.path.join(LOGICALS_DIR, f"{code}_LZ.npy")
        np.save(path, np.ascontiguousarray(LZ, dtype=np.uint8))
        lines.append(f"{_sha256(path)}  {code}_LZ.npy\n")
    with open(os.path.join(LOGICALS_DIR, "SHA256SUMS"), "w") as fh:
        fh.writelines(lines)


@dataclass
class FrozenSet:
    """One frozen validation set. `frames` is what the decoder sees (one row per shot).
    Arm A: `obs` (observable flips); Arms B/C: `truth` (data X error per shot, length n),
    `LZ` (published basis) and `labels` = L_Z truth mod 2 (the logical-label convention)."""
    arm: str
    name: str
    path: str
    frames: np.ndarray
    shots: int
    p: float
    d: int | None = None
    code: str | None = None
    obs: np.ndarray | None = None
    truth: np.ndarray | None = None
    LZ: np.ndarray | None = None
    labels: np.ndarray | None = None
    HZ: np.ndarray | None = None
    H_st: sp.csr_matrix | None = None
    channel_probs: np.ndarray | None = None
    T: int | None = None
    circuit: object = field(default=None, repr=False)


def surface_circuit(d: int):
    """Stim circuit of Arm A exactly as in verify_redecode.py (p = 1e-3, 25 rounds)."""
    import stim
    p = SURFACE_P
    return stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=SURFACE_ROUNDS,
                                  after_clifford_depolarization=p, before_measure_flip_probability=p,
                                  after_reset_flip_probability=p, before_round_data_depolarization=p)


def load_frozen(arm: str, *, d: int | None = None, code: str | None = None, p: float) -> FrozenSet:
    """Load a frozen set (sha256 checked against data/frozen_validation/SHA256SUMS).
    Arm A: load_frozen("A", d=5, p=0.001); Arm B: load_frozen("B", code="bb72", p=0.04);
    Arm C: load_frozen("C", p=0.02) (bb72, T = 6, q = p, the 2,000-shot set)."""
    arm = arm.upper()
    sums = os.path.join(FROZEN_DIR, "SHA256SUMS")
    if arm == "A":
        if d is None or p != SURFACE_P:
            raise ValueError("Arm A needs d and p=0.001")
        path = _one(f"surface_d{d}_p{p}_r{SURFACE_ROUNDS}_s100000_seed20260708_*.npz")
        _check_sum(path, sums)
        z = np.load(path)
        circ = surface_circuit(d)
        det, obs = z["det"].astype(np.uint8), z["obs"].astype(np.uint8)
        assert det.shape[1] == circ.num_detectors and obs.shape[1] == circ.num_observables
        return FrozenSet("A", os.path.basename(path)[:-4], path, det, det.shape[0], p, d=d,
                         obs=obs, circuit=circ)
    if arm == "B":
        if code not in ("bb72", "bb144"):
            raise ValueError("Arm B needs code='bb72' or 'bb144'")
        import prepare_qldpc as Q
        path = _one(f"qldpc_{code}_p{p}_s50000_seed20260708_*.npz")
        _check_sum(path, sums)
        z = np.load(path)
        err, syn = z["err"].astype(np.uint8), z["syn"].astype(np.uint8)
        _, HZ = Q.build_bb_code(code)
        assert np.array_equal(syn, ((err.astype(int) @ HZ.T.astype(int)) % 2).astype(np.uint8))
        LZ = load_logicals(code)
        labels = ((err.astype(int) @ LZ.T.astype(int)) % 2).astype(np.uint8)
        return FrozenSet("B", os.path.basename(path)[:-4], path, syn, syn.shape[0], p, code=code,
                         truth=err, LZ=LZ, labels=labels, HZ=HZ)
    if arm == "C":
        if code not in (None, "bb72"):
            raise ValueError("the frozen Arm C sets are bb72 only (T = 6)")
        import prepare_qldpc_phenom as PH
        path = _one(f"phenom_bb72_T{PHENOM_T}_p{p}_q{p}_s2000_seed20260709_*.npz")
        _check_sum(path, sums)
        z = np.load(path)
        det, etot = z["det"].astype(np.uint8), z["etot"].astype(np.uint8)
        _, HZ = PH.build_bb_code("bb72")
        m, n = HZ.shape
        H_st = PH.build_spacetime_check(HZ, PHENOM_T)
        assert det.shape[1] == (PHENOM_T + 1) * m
        LZ = load_logicals("bb72")
        labels = ((etot.astype(int) @ LZ.T.astype(int)) % 2).astype(np.uint8)
        chan = np.array([p] * (PHENOM_T * n) + [p] * (PHENOM_T * m))
        return FrozenSet("C", os.path.basename(path)[:-4], path, det, det.shape[0], p, code="bb72",
                         truth=etot, LZ=LZ, labels=labels, HZ=HZ, H_st=H_st,
                         channel_probs=chan, T=PHENOM_T)
    raise ValueError(f"unknown arm {arm!r}")


# --------------------------------------------------------------------------- scoring
def _decode_all(decoder, frames, batch):
    if hasattr(decoder, "decode_batch"):
        if batch is None:
            return np.asarray(decoder.decode_batch(frames))
        parts = [np.asarray(decoder.decode_batch(frames[i:i + batch]))
                 for i in range(0, frames.shape[0], int(batch))]
        return np.concatenate(parts, axis=0)
    outs = [np.asarray(decoder.decode(f)).ravel() for f in frames]
    lens = {o.size for o in outs}
    if len(lens) != 1:
        raise ValueError(f"decoder returned outputs of different lengths {sorted(lens)}")
    return np.stack(outs)


def failures_from_output(out, data: FrozenSet) -> tuple[np.ndarray, str]:
    """Per-shot failure vector from a 2-D output array (shots x length); see module docstring.
    Returns (fail, kind) with kind in {'logical', 'correction', 'spacetime_correction'}."""
    out = np.asarray(out)
    if out.ndim == 1:
        out = out[:, None]
    if out.shape[0] != data.shots:
        raise ValueError(f"{out.shape[0]} outputs for {data.shots} shots")
    if not np.isin(out, (0, 1)).all():
        raise ValueError("decoder output must be 0/1 valued")
    out = out.astype(np.uint8)
    L = out.shape[1]
    if data.arm == "A":
        if L != 1:
            raise ValueError(f"Arm A expects length 1 (observable prediction), got {L}")
        return out[:, 0] != data.obs[:, 0], "logical"
    n, k = data.truth.shape[1], data.LZ.shape[0]
    if L == k:
        return (out != data.labels).any(axis=1), "logical"
    if L == n:
        corr, kind = out, "correction"
    elif data.arm == "C" and L == data.T * n + data.T * data.HZ.shape[0]:
        corr = np.bitwise_xor.reduce(out[:, :data.T * n].reshape(-1, data.T, n), axis=1)
        kind = "spacetime_correction"
    else:
        raise ValueError(f"Arm {data.arm}: output length {L} is none of k={k}, n={n}"
                         + (f", T*n+T*m={data.T * n + data.T * data.HZ.shape[0]}" if data.arm == "C" else ""))
    resid = (data.truth ^ corr).astype(int)
    return ((resid @ data.LZ.T.astype(int)) % 2).any(axis=1), kind


def score(decoder, data: FrozenSet, batch: int | None = None) -> dict:
    """Decode every shot of `data` and score it.
    If the decoder has decode_batch, it is called once on all frames (batch=None) or on
    consecutive chunks of `batch` frames; otherwise decode(frame) is called once per shot.
    Returns failures f, shots N, ler = f/N, ci95 (exact Clopper-Pearson, equal-tailed),
    output kind, and `fail`, the boolean per-shot failure vector (input to compare())."""
    out = _decode_all(decoder, data.frames, batch)
    fail, kind = failures_from_output(out, data)
    f, n = int(fail.sum()), int(fail.size)
    return {"set": data.name, "failures": f, "shots": n, "ler": f / n,
            "ci95": clopper_pearson(f, n), "kind": kind, "fail": fail}


def save_failures(path: str, fail) -> None:
    """Save a failure vector as np.packbits of the boolean vector (the compare.py format)."""
    np.save(path, np.packbits(np.asarray(fail).astype(bool).ravel()))


def load_failures(path: str, shots: int) -> np.ndarray:
    """Inverse of save_failures; `shots` is the unpadded length."""
    packed = np.load(path)
    if packed.dtype != np.uint8 or packed.ndim != 1 or packed.size != (shots + 7) // 8:
        raise ValueError(f"{path}: expected {(shots + 7) // 8} packed uint8 bytes for {shots} shots")
    bits = np.unpackbits(packed)
    if bits[shots:].any():
        raise ValueError(f"{path}: nonzero padding bits; wrong --shots?")
    return bits[:shots].astype(bool)


# --------------------------------------------------------------------------- reference decoders
class ReferenceMWPM:
    """Arm A reference: PyMatching on the Stim DEM (decompose_errors=True) of surface_circuit(d).
    decode / decode_batch return the predicted flip of observable 0 (length 1 per shot)."""

    def __init__(self, d: int):
        import pymatching
        dem = surface_circuit(d).detector_error_model(decompose_errors=True)
        self.matching = pymatching.Matching.from_detector_error_model(dem)

    def decode(self, frame):
        return np.asarray(self.matching.decode(frame), dtype=np.uint8)

    def decode_batch(self, frames):
        return np.asarray(self.matching.decode_batch(frames), dtype=np.uint8)


class ReferenceBPOSD:
    """Arms B/C reference: ldpc decoder on parity-check matrix H with the paper's settings
    (min-sum, ms_scaling_factor 0.625, serial schedule, max_iter 50).
    osd='osd7': BpOsdDecoder, OSD_CS order 7 (the paper's reference);
    osd='osd0': BpOsdDecoder, OSD_0;  osd='bp': BpDecoder (no OSD).
    Give error_rate (Arm B) or channel_probs (Arm C, with H = H_st). decode returns the
    physical correction (length H.shape[1]: n for Arm B, T*n+T*m for Arm C)."""

    def __init__(self, H, error_rate: float | None = None, channel_probs=None, osd: str = "osd7"):
        from ldpc import BpDecoder, BpOsdDecoder
        if (error_rate is None) == (channel_probs is None):
            raise ValueError("give exactly one of error_rate, channel_probs")
        prior = {"error_rate": error_rate} if error_rate is not None else \
            {"channel_probs": list(np.asarray(channel_probs, dtype=float))}
        H = sp.csr_matrix(np.asarray(H.toarray() if sp.issparse(H) else H, dtype=np.uint8))
        if osd == "osd7":
            self.dec = BpOsdDecoder(H, **prior, **BP_KW, osd_method="OSD_CS", osd_order=7)
        elif osd == "osd0":
            self.dec = BpOsdDecoder(H, **prior, **BP_KW, osd_method="OSD_0", osd_order=0)
        elif osd == "bp":
            self.dec = BpDecoder(H, **prior, **BP_KW)
        else:
            raise ValueError(f"osd must be 'osd7', 'osd0' or 'bp', got {osd!r}")
        self.osd = osd

    def decode(self, frame):
        return np.asarray(self.dec.decode(np.asarray(frame, dtype=np.uint8)), dtype=np.uint8)


def reference_for(data: FrozenSet, osd: str = "osd7"):
    """The paper's reference decoder for a loaded set (MWPM for A, BP-OSD otherwise)."""
    if data.arm == "A":
        return ReferenceMWPM(data.d)
    if data.arm == "B":
        return ReferenceBPOSD(data.HZ, error_rate=data.p, osd=osd)
    return ReferenceBPOSD(data.H_st, channel_probs=data.channel_probs, osd=osd)


def fmt_result(r: dict) -> str:
    lo, hi = r["ci95"]
    return (f"{r['set']}: {r['failures']}/{r['shots']}  LER={r['ler']:.4e}  "
            f"95% CP [{lo:.4e}, {hi:.4e}]  ({r['kind']})")


def fmt_compare(c: dict) -> str:
    ci = c["ci95_b_share"]
    cis = "None (b+c=0)" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"
    ps = "0 (below double-precision range)" if c["p_value"] == 0.0 else f"{c['p_value']:.4g}"
    return (f"b={c['b']} c={c['c']} N={c['N']} delta={c['delta']:+.6f} "
            f"McNemar exact p={ps}  b/(b+c) 95% CP {cis}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--export-logicals"]:
        export_logicals()
        print(open(os.path.join(LOGICALS_DIR, "SHA256SUMS")).read(), end="")
    else:
        print(__doc__)
