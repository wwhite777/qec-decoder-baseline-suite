"""
P1 prepare_qldpc_phenom.py — FROZEN contract for the qLDPC (bivariate-bicycle) track
under **PHENOMENOLOGICAL noise** (multi-round, space-time decoding).

This is the honest v2 that goes *beyond* code-capacity (prepare_qldpc.py) toward realistic
fault-tolerant decoding. Instead of a single noiseless shot of the syndrome, we run
**T rounds of NOISY syndrome extraction**:
  * every round, each data qubit picks up an i.i.d. X error at rate  p   (Bernoulli),
  * every round, each Z-check measurement outcome is flipped with prob  q  (measurement /
    readout error), and
  * after the T noisy rounds we do ONE final PERFECT (noiseless) stabiliser readout so the
    last round's measurement errors and late data errors become detectable (standard
    phenomenological boundary condition).
The decoder sees only the noisy per-round syndromes; it must infer the *total accumulated*
data error and correct it. Logical failure iff the residual anticommutes with a Z-logical.
This is strictly harder than code-capacity and much closer to reality (measurement noise +
time is what makes real QEC decoding hard); circuit-level noise (a full Stim BB
syndrome-extraction CIRCUIT with the ancilla schedule + hook errors) is the further step and
is intentionally NOT modelled here — see NOTE.

Space-time (detector) decoding — the standard phenomenological construction (single sector,
Z-checks HZ detecting X data errors; by CSS symmetry the X sector is analogous):
  variables   e_r in F2^n   (X data error introduced before round r's measurement, r=0..T-1)
              f_r in F2^m   (flip of round r's measured syndrome,                  r=0..T-1)
  measured    s_r = HZ @ (e_0 ^ ... ^ e_r) ^ f_r                                   r=0..T-1
  final perfect readout      s_T = HZ @ (e_0 ^ ... ^ e_{T-1})   (no f)
  DETECTORS (XOR of consecutive syndromes; a detector = 0 with no noise):
     D_0 = s_0                = HZ e_0 ^ f_0
     D_r = s_r ^ s_{r-1}      = HZ e_r ^ f_r ^ f_{r-1}          r=1..T-1
     D_T = s_T ^ s_{T-1}      = f_{T-1}                          (final perfect round)
  => (T+1)*m detector bits.  Unknowns ordered  [e_0..e_{T-1} | f_0..f_{T-1}]  ((T*n)+(T*m) cols).
  H_st is the (T+1)*m  x  (T*n + T*m)  sparse space-time parity-check matrix built below;
  BP-OSD decodes it with a per-column channel prior (p on the e-columns, q on the f-columns).
  The physical correction is  Ê = XOR_r ê_r  (total data error); residual R = E_true ^ Ê;
  **logical failure iff L_Z R != 0 (mod 2)** — same L_Z basis as the code-capacity contract.

FROZEN like prepare_qldpc.py: do not edit when adding a decoder. It owns the space-time
construction, the pinned (T, p-sweep, q, seed, shot count) and evaluate_ler_phenom(decode_fn).
Reuses the BB code construction + clean CSS logicals from prepare_qldpc.py (single source of
truth for H_X/H_Z/L_Z) — this file adds ONLY the multi-round noise + space-time detector layer.

Open-source: numpy, scipy, ldpc (BP-OSD + mod2), Roffe et al.
"""
from __future__ import annotations
import os, hashlib, time
import numpy as np
import scipy.sparse as sp

# reuse the FROZEN BB code construction + clean CSS logicals (do NOT duplicate the algebra)
import prepare_qldpc as Q
from prepare_qldpc import build_bb_code, css_logicals, code_params, code_str, BB_CODES

# --- FROZEN pinned phenomenological config --------------------------------
# T rounds of noisy extraction ~ the code distance d (the usual fault-tolerant choice):
ROUNDS = {"bb72": 6, "bb144": 12}     # T = d for each BB code
DEFAULT_CODE = "bb72"
P_SWEEP = [0.02, 0.04, 0.06, 0.08]    # data-qubit X-error rate per round (matches code-capacity sweep)
Q_EQ_P = True                          # measurement-flip rate q := p (balanced phenomenological model)
P_REF = 0.02                           # reference p for the results.tsv scalar
# rarer events per shot than code-capacity? no — MORE noise, so fewer shots still resolve LER,
# but keep a healthy count and always report failure count + Wilson bound.
DEFAULT_SHOTS = 20_000
SEED = 20260709                        # pinned so the phenom val set is a frozen artifact
PRIMARY_METRIC = "logical_error_rate"
PRIMARY_DIRECTION = "lower"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ququ_p1_qec")
os.makedirs(CACHE_DIR, exist_ok=True)


def q_for(p: float) -> float:
    """Measurement-flip rate for a given data rate p (frozen policy: q = p)."""
    return p if Q_EQ_P else p


# --- space-time (detector) parity-check matrix ----------------------------
def build_spacetime_check(HZ: np.ndarray, T: int) -> sp.csr_matrix:
    """(T+1)*m  x  (T*n + T*m) sparse detector check matrix H_st for the Z-sector.
    Column order: [e_0..e_{T-1} (each n) | f_0..f_{T-1} (each m)].
    Row block d (d=0..T) is detector D_d as derived in the module docstring."""
    m, n = HZ.shape
    HZs = sp.csr_matrix(HZ.astype(np.uint8))
    Im = sp.identity(m, dtype=np.uint8, format="csr")
    Z_mn = sp.csr_matrix((m, n), dtype=np.uint8)
    Z_mm = sp.csr_matrix((m, m), dtype=np.uint8)
    rowblocks = []
    for d in range(T + 1):
        eparts = [HZs if (d < T and r == d) else Z_mn for r in range(T)]
        fparts = []
        for r in range(T):
            coeff = Z_mm
            if d < T:
                if r == d:
                    coeff = Im            # f_d
                if d >= 1 and r == d - 1:
                    coeff = Im            # f_{d-1}
            else:                          # d == T : final perfect round -> only f_{T-1}
                if r == T - 1:
                    coeff = Im
            fparts.append(coeff)
        rowblocks.append(sp.hstack(eparts + fparts, format="csr"))
    return sp.vstack(rowblocks, format="csr")


def channel_prior(n: int, m: int, T: int, p: float, q: float) -> np.ndarray:
    """Per-column BP prior over [e_0..e_{T-1} | f_0..f_{T-1}]: p on data cols, q on meas cols."""
    return np.concatenate([np.full(T * n, max(p, 1e-12)),
                           np.full(T * m, max(q, 1e-12))])


# --- frozen phenomenological dataset --------------------------------------
def generate_dataset(code: str = DEFAULT_CODE, p: float = P_REF,
                     shots: int = DEFAULT_SHOTS) -> dict:
    """FROZEN phenomenological space-time dataset (cached npz keyed by config+seed).
    Returns dict with:
        detectors  : uint8 [shots, (T+1)*m]   XOR-of-consecutive-syndrome detector bits
        err_total  : uint8 [shots, n]         ground-truth TOTAL data X-error e_0^...^e_{T-1}
        HZ, LZ     : Z parity-check + clean Z-logical basis (k x n)
        H_st       : sparse space-time detector check matrix
        T, q       : rounds and measurement-flip rate used
    """
    T = ROUNDS[code]
    q = q_for(p)
    HX, HZ = build_bb_code(code)
    _, LZ = css_logicals(HX, HZ)
    m, n = HZ.shape
    H_st = build_spacetime_check(HZ, T)

    key = f"phenom_{code}_T{T}_p{p}_q{q}_s{shots}_seed{SEED}"
    tag = hashlib.md5(key.encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{key}_{tag}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {"detectors": z["det"], "err_total": z["etot"],
                "HZ": HZ, "LZ": LZ, "H_st": H_st, "T": T, "q": q}

    rng = np.random.default_rng(SEED)
    HZi = HZ.astype(int)
    dets = np.zeros((shots, (T + 1) * m), dtype=np.uint8)
    etot = np.zeros((shots, n), dtype=np.uint8)
    for s in range(shots):
        e = (rng.random((T, n)) < p).astype(np.uint8)      # data errors per round
        f = (rng.random((T, m)) < q).astype(np.uint8)      # measurement flips per round
        cum = np.zeros(n, dtype=int)
        s_meas = np.zeros((T, m), dtype=np.uint8)
        for r in range(T):
            cum = (cum + e[r]) % 2
            s_meas[r] = ((HZi @ cum) % 2).astype(np.uint8) ^ f[r]
        s_perfect = ((HZi @ cum) % 2).astype(np.uint8)     # final noiseless readout
        D = np.zeros((T + 1, m), dtype=np.uint8)
        D[0] = s_meas[0]
        for r in range(1, T):
            D[r] = s_meas[r] ^ s_meas[r - 1]
        D[T] = s_perfect ^ s_meas[T - 1]
        dets[s] = D.reshape(-1)
        etot[s] = cum.astype(np.uint8)
    np.savez_compressed(path, det=dets, etot=etot)
    return {"detectors": dets, "err_total": etot,
            "HZ": HZ, "LZ": LZ, "H_st": H_st, "T": T, "q": q}


def _logical_failures(err_total, corr_total, LZ) -> int:
    """# shots whose residual (E_true ^ Ê) anticommutes with any Z-logical."""
    resid = (err_total.astype(np.uint8) ^ corr_total.astype(np.uint8))
    la = (resid.astype(int) @ LZ.T.astype(int)) % 2
    return int(np.any(la, axis=1).sum())


def evaluate_ler_phenom(decode_fn, code: str = DEFAULT_CODE, p: float = P_REF,
                        shots: int = DEFAULT_SHOTS) -> dict:
    """GROUND-TRUTH phenomenological (multi-round) LER on the frozen val set.
    `decode_fn(detectors[shots, (T+1)*m]) -> total data correction Ê [shots, n]` (uint8):
    the decoder must return the *net* physical X-error correction on the n data qubits.
    Returns LER, raw failure count, shots, and the noise descriptor (T, p, q)."""
    ds = generate_dataset(code, p, shots)
    det, etot, LZ = ds["detectors"], ds["err_total"], ds["LZ"]
    corr = np.asarray(decode_fn(det)).reshape(etot.shape).astype(np.uint8)
    fails = _logical_failures(etot, corr, LZ)
    ler = fails / len(etot)
    return {"logical_error_rate": ler, "failures": fails, "shots": len(etot),
            "p": p, "q": ds["q"], "T": ds["T"], "code": code}


def measure_latency_us(decode_fn, code: str = DEFAULT_CODE, p: float = P_REF,
                       shots: int = 2_000) -> float:
    """Real steady-state decode latency (wall-clock), us PER SHOT (a shot = T-round block).
    NOTE: this is per multi-round shot, NOT per round; divide by T for a per-round figure."""
    ds = generate_dataset(code, p, shots)
    det = ds["detectors"]
    decode_fn(det[:64])                  # warm up
    t0 = time.time()
    decode_fn(det)
    return (time.time() - t0) / len(det) * 1e6


if __name__ == "__main__":
    # self-check on import: shapes, ranks, and the T=1/q=0 -> code-capacity sanity note
    from ldpc import mod2
    for c in BB_CODES:
        HX, HZ = build_bb_code(c)
        T = ROUNDS[c]
        m, n = HZ.shape
        H_st = build_spacetime_check(HZ, T)
        r = mod2.rank(H_st)
        print(f"{c}: {code_str(c)}  T={T}  H_st={H_st.shape}  rank={r}  "
              f"n={n} m={m}  det_bits={(T+1)*m}  err_vars={T*n + T*m}")
