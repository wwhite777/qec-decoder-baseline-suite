"""
P1 prepare_qldpc.py — FROZEN contract for the qLDPC (bivariate-bicycle) track.

This mirrors prepare.py's style but for a **bivariate-bicycle (BB) code** under
**code-capacity noise** (data-qubit i.i.d. bit-flip errors at physical rate p, a single
shot of the syndrome). Code-capacity is the v1 battlefield where a neural decoder can be
compared against BP-OSD; **circuit-level noise (Stim + a detector-error-model over
syndrome-extraction rounds) is the explicit next step** and is intentionally NOT modelled
here (see NOTE below).

Why BB / qLDPC: on surface codes MWPM is a strong graphlike reference (prepare.py + baseline_mwpm.py),
so the neural edge must come from a code family where MWPM does not apply. BB codes are
the leading low-overhead qLDPC memory (Bravyi et al., "High-threshold and low-overhead
fault-tolerant quantum memory", Nature 627 (2024); arXiv:2308.07915). The classical
reference decoder here is **BP-OSD** (Panteleev & Kalachev; Roffe et al., `ldpc` package).

Construction (Z_l x Z_m, cyclic shift matrices  x = S_l ⊗ I_m,  y = I_l ⊗ S_m):
    A = x^3 + y + y^2 ,   B = y^3 + x + x^2   (mod 2)
    H_X = [ A | B ] ,     H_Z = [ B^T | A^T ]
    n = 2*l*m physical qubits, k = n - rank(H_X) - rank(H_Z) logical qubits.
  * l=6,  m=6  -> [[72, 12, 6]]   (verified here: n=72, k=12, CSS commute OK)
  * l=12, m=6  -> [[144,12,12]]  gross code (verified: n=144, k=12, CSS commute OK)
  d (6 / 12) are the published literature values for these exact polynomials; n, k and the
  CSS commutation H_X H_Z^T = 0 are checked directly at import via code_params().

Noise / task (code-capacity, single-shot):
  * i.i.d. X error e ~ Bernoulli(p)^n on the data qubits.
  * Z-check syndrome  s = H_Z e  (mod 2)  detects X errors.
  * A decoder proposes ê; residual r = e ⊕ ê.
  * **Logical failure iff r anticommutes with some Z-logical**, i.e. L_Z r ≠ 0 (mod 2),
    where L_Z is a clean basis of ker(H_X) / rowspace(H_Z) (exactly k rows).
  (We decode the X-error / Z-check sector; by CSS symmetry the Z-error sector is analogous.)

FROZEN like prepare.py: do not edit when adding a decoder. It owns code construction, the
pinned p-sweep + seed + shot count, and evaluate_ler(decode_fn).

Open-source: numpy, scipy, ldpc (BP-OSD + mod2 linear algebra), Roffe et al.
"""
from __future__ import annotations
import os, hashlib, time
import numpy as np
import scipy.sparse as sp

# --- FROZEN pinned qLDPC config -------------------------------------------
BB_CODES = {
    # name -> (l, m)  with the shared A,B polynomial family below
    "bb72":  (6, 6),    # [[72, 12, 6]]
    "bb144": (12, 6),   # [[144,12,12]] gross code
}
DEFAULT_CODE = "bb72"
# A = x^3 + y + y^2 ; B = y^3 + x + x^2   (('x'|'y', power) terms, summed mod 2)
A_TERMS = [("x", 3), ("y", 1), ("y", 2)]
B_TERMS = [("y", 3), ("x", 1), ("x", 2)]

P_SWEEP = [0.02, 0.04, 0.06, 0.08]   # pinned physical error-rate sweep (code-capacity)
P_REF = 0.02                          # reference p for the results.tsv scalar
DEFAULT_SHOTS = 50_000                # per-p shots (report failure count too; low-p bb144
                                      # is a rare-event point -> read its Wilson 95%hi bound)
SEED = 20260708                       # pinned so the val set is a frozen artifact
PRIMARY_METRIC = "logical_error_rate"
PRIMARY_DIRECTION = "lower"

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "ququ_p1_qec")
os.makedirs(CACHE_DIR, exist_ok=True)


# --- BB code construction --------------------------------------------------
def _cyclic_shift(n: int) -> np.ndarray:
    """S_n : n x n cyclic-shift permutation matrix, S[i, (i+1) mod n] = 1."""
    S = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        S[i, (i + 1) % n] = 1
    return S


def _poly(terms, x, y, dim):
    M = np.zeros((dim, dim), dtype=int)
    for var, pw in terms:
        base = x if var == "x" else y
        M = (M + np.linalg.matrix_power(base.astype(int), pw)) % 2
    return M.astype(np.uint8)


def build_bb_code(code: str = DEFAULT_CODE):
    """Return (H_X, H_Z) as dense uint8 arrays for the named BB code."""
    l, m = BB_CODES[code]
    Il, Im = np.eye(l, dtype=np.uint8), np.eye(m, dtype=np.uint8)
    x = np.kron(_cyclic_shift(l), Im)     # (l*m) x (l*m)
    y = np.kron(Il, _cyclic_shift(m))
    A = _poly(A_TERMS, x, y, l * m)
    B = _poly(B_TERMS, x, y, l * m)
    HX = (np.hstack([A, B]) % 2).astype(np.uint8)      # (l*m) x (2*l*m)
    HZ = (np.hstack([B.T, A.T]) % 2).astype(np.uint8)
    return HX, HZ


def _rank(M) -> int:
    from ldpc import mod2
    return mod2.rank(sp.csr_matrix(np.asarray(M, np.uint8)))


def _quotient(ker_rows: np.ndarray, stab: np.ndarray) -> np.ndarray:
    """Rows of ker_rows kept iff they raise rank over rowspace(stab): the logical reps."""
    from ldpc import mod2
    cur = [r for r in np.asarray(stab, np.uint8)]
    cur_rank = mod2.rank(sp.csr_matrix(np.array(cur, np.uint8))) if cur else 0
    logs = []
    for r in ker_rows:
        trial = np.array(cur + [r], dtype=np.uint8)
        nr = mod2.rank(sp.csr_matrix(trial))
        if nr > cur_rank:
            cur.append(r)
            cur_rank = nr
            logs.append(r)
    return np.array(logs, dtype=np.uint8)


def css_logicals(HX: np.ndarray, HZ: np.ndarray):
    """Clean CSS logical bases (each exactly k x n):
       L_Z = ker(H_X) / rowspace(H_Z) ,  L_X = ker(H_Z) / rowspace(H_X)."""
    from ldpc import mod2
    ker_HX = mod2.kernel(sp.csr_matrix(HX)).toarray().astype(np.uint8)
    ker_HZ = mod2.kernel(sp.csr_matrix(HZ)).toarray().astype(np.uint8)
    LZ = _quotient(ker_HX, HZ)
    LX = _quotient(ker_HZ, HX)
    return LX, LZ


def code_params(HX: np.ndarray, HZ: np.ndarray) -> dict:
    """n, k, ranks, and the CSS commutation check H_X H_Z^T == 0 (mod 2)."""
    n = HX.shape[1]
    rx, rz = _rank(HX), _rank(HZ)
    css_ok = int(((HX.astype(int) @ HZ.T.astype(int)) % 2).sum()) == 0
    return {"n": n, "k": n - rx - rz, "rank_HX": rx, "rank_HZ": rz, "css_commute": css_ok}


# distance is the published literature value for these exact polynomials (Nature 627, 2024);
# n, k, CSS-commutation are verified directly above. We do NOT recompute the min-weight
# logical (needs ILP/GAP search) — report d as literature and say so.
LITERATURE_D = {"bb72": 6, "bb144": 12}


def code_str(code: str = DEFAULT_CODE) -> str:
    HX, HZ = build_bb_code(code)
    prm = code_params(HX, HZ)
    return f"[[{prm['n']},{prm['k']},{LITERATURE_D[code]}]]"


# --- frozen code-capacity dataset -----------------------------------------
def generate_dataset(code: str = DEFAULT_CODE, p: float = P_REF,
                     shots: int = DEFAULT_SHOTS) -> dict:
    """FROZEN code-capacity syndrome dataset (cached npz keyed by config+seed).
    Returns dict with:
        errors     : uint8 [shots, n]   i.i.d. X errors ~ Bernoulli(p)
        syndromes  : uint8 [shots, m_Z]  = H_Z @ errors  (mod 2)
        HX, HZ     : parity-check matrices
        LX, LZ     : clean CSS logical bases (k x n)
    """
    key = f"qldpc_{code}_p{p}_s{shots}_seed{SEED}"
    tag = hashlib.md5(key.encode()).hexdigest()[:10]
    path = os.path.join(CACHE_DIR, f"{key}_{tag}.npz")
    HX, HZ = build_bb_code(code)
    LX, LZ = css_logicals(HX, HZ)
    n = HX.shape[1]
    if os.path.exists(path):
        z = np.load(path)
        return {"errors": z["err"], "syndromes": z["syn"],
                "HX": HX, "HZ": HZ, "LX": LX, "LZ": LZ}
    rng = np.random.default_rng(SEED)
    err = (rng.random((shots, n)) < p).astype(np.uint8)
    syn = (err @ HZ.T.astype(int) % 2).astype(np.uint8)   # [shots, m_Z]
    np.savez_compressed(path, err=err, syn=syn)
    return {"errors": err, "syndromes": syn,
            "HX": HX, "HZ": HZ, "LX": LX, "LZ": LZ}


def _logical_failures(errors, preds, LZ) -> int:
    """# shots whose residual (e ⊕ ê) anticommutes with any Z-logical."""
    resid = (errors.astype(np.uint8) ^ preds.astype(np.uint8))
    la = (resid.astype(int) @ LZ.T.astype(int)) % 2          # [shots, k]
    return int(np.any(la, axis=1).sum())


def evaluate_ler(decode_fn, code: str = DEFAULT_CODE, p: float = P_REF,
                 shots: int = DEFAULT_SHOTS) -> dict:
    """GROUND-TRUTH code-capacity LER on the frozen val set.
    `decode_fn(syndromes[shots,m_Z]) -> predicted errors ê [shots, n]` (uint8).
    Returns LER, raw failure count and shots (for statistics)."""
    ds = generate_dataset(code, p, shots)
    err, syn, LZ = ds["errors"], ds["syndromes"], ds["LZ"]
    pred = np.asarray(decode_fn(syn)).reshape(err.shape).astype(np.uint8)
    fails = _logical_failures(err, pred, LZ)
    ler = fails / len(err)
    return {"logical_error_rate": ler, "failures": fails, "shots": len(err),
            "p": p, "code": code}


def measure_latency_us(decode_fn, code: str = DEFAULT_CODE, p: float = P_REF,
                       shots: int = 5_000) -> float:
    """Real steady-state decode latency (wall-clock), us PER SHOT (single-shot code-capacity)."""
    ds = generate_dataset(code, p, shots)
    syn = ds["syndromes"]
    decode_fn(syn[:256])                 # warm up
    t0 = time.time()
    decode_fn(syn)
    return (time.time() - t0) / len(syn) * 1e6


if __name__ == "__main__":
    # self-check on import: print params for both codes
    for c in BB_CODES:
        HX, HZ = build_bb_code(c)
        prm = code_params(HX, HZ)
        LX, LZ = css_logicals(HX, HZ)
        print(f"{c}: {code_str(c)}  {prm}  k(L_X)={LX.shape[0]} k(L_Z)={LZ.shape[0]}")
