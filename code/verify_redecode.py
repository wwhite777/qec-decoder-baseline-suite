"""Re-decode every frozen validation set and recompute every failure count and exact
Clopper-Pearson interval in the paper, then compare the counts with results/.

The validation arrays are shipped in data/frozen_validation/ (Stim's seeded sampling is not
consistent across Stim versions, so the arrays themselves are the frozen artifact). The BB
parity-check matrices, the logical-Z basis (own GF(2) elimination) and the space-time detector
matrix are rebuilt here from their definitions and checked against code/prepare_qldpc*.py.

Usage (CPU, about one minute):
    python code/verify_redecode.py [out.json]
Exit 0 only if all 16 failure counts equal the frozen CSV values.
Tested with stim 1.16.0, pymatching 2.4.0, ldpc 2.4.1, numpy 2.2.6, scipy 1.15.3.
"""
import csv, glob, json, os, sys, time
import numpy as np
import scipy.sparse as sp
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "frozen_validation")
RES = os.path.join(ROOT, "results")
BP = dict(max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625,
          schedule="serial", osd_method="OSD_CS", osd_order=7)


def cp(f, n, a=0.05):
    lo = 0.0 if f == 0 else float(beta.ppf(a / 2, f, n - f + 1))
    hi = 1.0 if f == n else float(beta.ppf(1 - a / 2, f + 1, n - f))
    return lo, hi


def one(pattern):
    hits = sorted(glob.glob(os.path.join(DATA, pattern)))
    if len(hits) != 1:
        raise SystemExit(f"data lookup {pattern}: {len(hits)} files")
    return hits[0]


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def rref_gf2(M):
    M = (np.asarray(M, dtype=np.uint8) % 2).copy()
    rows, cols = M.shape
    piv, r = [], 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.nonzero(M[r:, c])[0]
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            M[[r, p]] = M[[p, r]]
        others = np.nonzero(M[:, c])[0]
        M[others[others != r]] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r], piv


def rank_gf2(M):
    return rref_gf2(M)[0].shape[0]


def kernel_gf2(M):
    R, piv = rref_gf2(M)
    n = M.shape[1]
    basis = []
    for fcol in [c for c in range(n) if c not in set(piv)]:
        v = np.zeros(n, dtype=np.uint8)
        v[fcol] = 1
        for i, pc in enumerate(piv):
            v[pc] = R[i, fcol]
        basis.append(v)
    return np.array(basis, dtype=np.uint8)


def logical_basis(ker, stab):
    cur = np.asarray(stab, dtype=np.uint8)
    r0 = rank_gf2(cur)
    logs = []
    for v in ker:
        trial = np.vstack([cur, v[None, :]])
        r1 = rank_gf2(trial)
        if r1 > r0:
            cur, r0 = trial, r1
            logs.append(v)
    return np.array(logs, dtype=np.uint8)


def shift(n):
    S = np.zeros((n, n), dtype=np.uint8)
    S[np.arange(n), (np.arange(n) + 1) % n] = 1
    return S


def bb_code(l, m):
    x = np.kron(shift(l), np.eye(m, dtype=np.uint8))
    y = np.kron(np.eye(l, dtype=np.uint8), shift(m))
    mp = lambda M, k: np.linalg.matrix_power(M.astype(np.int64), k) % 2
    A = (mp(x, 3) + mp(y, 1) + mp(y, 2)) % 2
    B = (mp(y, 3) + mp(x, 1) + mp(x, 2)) % 2
    return np.hstack([A, B]).astype(np.uint8), np.hstack([B.T, A.T]).astype(np.uint8)


def spacetime(HZ, T):
    m, n = HZ.shape
    H = np.zeros(((T + 1) * m, T * n + T * m), dtype=np.uint8)
    I = np.eye(m, dtype=np.uint8)
    for d in range(T):
        H[d * m:(d + 1) * m, d * n:(d + 1) * n] = HZ
        H[d * m:(d + 1) * m, T * n + d * m:T * n + (d + 1) * m] = I
        if d >= 1:
            H[d * m:(d + 1) * m, T * n + (d - 1) * m:T * n + d * m] = I
    H[T * m:(T + 1) * m, T * n + (T - 1) * m:T * n + T * m] = I
    return H


def main():
    import stim, pymatching, ldpc
    from ldpc import BpOsdDecoder
    sys.path.insert(0, HERE)
    import prepare_qldpc as Q
    import prepare_qldpc_phenom as PH
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "verify_redecode.json")
    res = {"versions": {"stim": stim.__version__, "pymatching": pymatching.__version__,
                        "ldpc": ldpc.__version__, "numpy": np.__version__},
           "armA": [], "armB": [], "armC": [], "mismatches": []}
    t0 = time.time()

    frozen = {int(r["distance"]): int(r["failures"]) for r in read_csv(os.path.join(RES, "exp1", "p1_mwpm_baseline.csv"))}
    for d in (5, 7, 9, 11):
        z = np.load(one(f"surface_d{d}_p0.001_r25_s100000_seed20260708_*.npz"))
        det, obs = z["det"], z["obs"]
        circ = stim.Circuit.generated("surface_code:rotated_memory_z", distance=d, rounds=25,
                                      after_clifford_depolarization=1e-3,
                                      before_measure_flip_probability=1e-3,
                                      after_reset_flip_probability=1e-3,
                                      before_round_data_depolarization=1e-3)
        assert det.shape[1] == circ.num_detectors and obs.shape[1] == circ.num_observables
        mt = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
        wrong = mt.decode_batch(det)[:, 0].astype(bool) != obs[:, 0].astype(bool)
        f, n = int(wrong.sum()), len(obs)
        res["armA"].append({"d": d, "shots": n, "failures": f, "frozen_failures": frozen[d],
                            "ler": f / n, "cp95": cp(f, n)})
        if f != frozen[d]:
            res["mismatches"].append(f"armA d={d}: {f} vs {frozen[d]}")
        print(f"Arm A  d={d:<2}  {f}/{n}  (frozen {frozen[d]})", flush=True)

    frozenB = {(r["code"], float(r["p"])): int(r["failures"]) for r in read_csv(os.path.join(RES, "exp3", "p1_qldpc_bposd.csv"))}
    LZs, HZs = {}, {}
    for code, (l, m) in {"bb72": (6, 6), "bb144": (12, 6)}.items():
        HX, HZ = bb_code(l, m)
        HXq, HZq = Q.build_bb_code(code)
        assert np.array_equal(HX, HXq) and np.array_equal(HZ, HZq)
        LZ = logical_basis(kernel_gf2(HX), HZ)
        assert LZ.shape[0] == HX.shape[1] - rank_gf2(HX) - rank_gf2(HZ) == 12
        LZs[code], HZs[code] = LZ, HZ
        for p in (0.02, 0.04, 0.06, 0.08):
            z = np.load(one(f"qldpc_{code}_p{p}_s50000_seed20260708_*.npz"))
            err, syn = z["err"].astype(np.uint8), z["syn"].astype(np.uint8)
            assert np.array_equal(syn, (err.astype(int) @ HZ.T.astype(int) % 2).astype(np.uint8))
            dec = BpOsdDecoder(sp.csr_matrix(HZ), error_rate=p, **BP)
            ehat = np.array([dec.decode(s) for s in syn], dtype=np.uint8)
            f = int((((err ^ ehat).astype(int) @ LZ.T.astype(int)) % 2).any(axis=1).sum())
            n = len(err)
            fz = frozenB[(code, p)]
            res["armB"].append({"code": code, "p": p, "shots": n, "failures": f, "frozen_failures": fz,
                                "ler": f / n, "cp95": cp(f, n)})
            if f != fz:
                res["mismatches"].append(f"armB {code} p={p}: {f} vs {fz}")
            print(f"Arm B  {code:<5} p={p}  {f}/{n}  (frozen {fz})", flush=True)

    frozenC = {float(r["p"]): int(r["failures"]) for r in read_csv(os.path.join(RES, "exp3", "p1_qldpc_phenom.csv"))}
    HZ, LZ, T = HZs["bb72"], LZs["bb72"], 6
    m, nq = HZ.shape
    Hst = spacetime(HZ, T)
    assert np.array_equal(Hst, PH.build_spacetime_check(HZ, T).toarray().astype(np.uint8))
    for p in (0.02, 0.04, 0.06, 0.08):
        z = np.load(one(f"phenom_bb72_T6_p{p}_q{p}_s2000_seed20260709_*.npz"))
        det, etot = z["det"].astype(np.uint8), z["etot"].astype(np.uint8)
        dec = BpOsdDecoder(sp.csr_matrix(Hst), channel_probs=[p] * (T * nq) + [p] * (T * m), **BP)
        corr = np.array([np.bitwise_xor.reduce(dec.decode(x)[:T * nq].reshape(T, nq).astype(np.uint8), axis=0)
                         for x in det], dtype=np.uint8)
        f = int((((etot ^ corr).astype(int) @ LZ.T.astype(int)) % 2).any(axis=1).sum())
        n = len(etot)
        res["armC"].append({"code": "bb72", "T": T, "p": p, "q": p, "shots": n, "failures": f,
                            "frozen_failures": frozenC[p], "ler": f / n, "cp95": cp(f, n)})
        if f != frozenC[p]:
            res["mismatches"].append(f"armC p={p}: {f} vs {frozenC[p]}")
        print(f"Arm C  bb72 T=6 p=q={p}  {f}/{n}  (frozen {frozenC[p]})", flush=True)

    res["elapsed_s"] = round(time.time() - t0, 1)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    npts = len(res["armA"]) + len(res["armB"]) + len(res["armC"])
    print(f"{npts} points re-decoded, {len(res['mismatches'])} mismatches -> {out}")
    sys.exit(1 if res["mismatches"] or npts != 16 else 0)


if __name__ == "__main__":
    main()
