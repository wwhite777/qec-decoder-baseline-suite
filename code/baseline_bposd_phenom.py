"""
P1 frozen qLDPC baseline: BP-OSD on the bivariate-bicycle code(s) under PHENOMENOLOGICAL
(multi-round, measurement-noisy) noise — the honest v2 that goes beyond code-capacity toward
realistic fault-tolerant decoding. This is the reference logical error rate the neural decoder
must beat in the multi-round setting, where a matching-based MWPM does not apply to BB codes
and even BP-OSD must reason over the *space-time* detector graph.

Decoder: ldpc.BpOsdDecoder (min-sum BP + Ordered-Statistics Decoding fallback) applied to the
(T+1)*m x (T*n + T*m) space-time detector check matrix H_st built in prepare_qldpc_phenom.py,
with a per-column channel prior (p on data-error columns, q on measurement-error columns).
The correction is projected to the net data-qubit X-error (XOR over rounds) and the residual is
checked against the SAME clean Z-logical basis L_Z as the code-capacity contract. CPU-only.

Model (frozen in prepare_qldpc_phenom.py):
  T = d rounds of noisy syndrome extraction (bb72: T=6, bb144: T=12), + 1 final PERFECT readout;
  each round: data X-error rate p per qubit, measurement-flip rate q = p per check.

Run (CPU only):
    python baseline_bposd_phenom.py            # bb72 (+ bb144)
    python baseline_bposd_phenom.py bb72       # single code

Writes: results/results.tsv row + results/exp3/p1_qldpc_phenom.csv (exact data).
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import scipy.sparse as sp
from ldpc import BpOsdDecoder

import prepare_qldpc_phenom as P

# Result directory: repo-root results/ by default; override with $P1_RESULT_DIR.
RESULT_ROOT = os.environ.get(
    "P1_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
RESULT_DIR = os.path.join(RESULT_ROOT, "exp3")
os.makedirs(RESULT_DIR, exist_ok=True)

# BP-OSD hyperparameters — IDENTICAL to the code-capacity baseline (fair comparison)
BP_MAX_ITER = 50
BP_METHOD = "minimum_sum"
MS_SCALE = 0.625
BP_SCHEDULE = "serial"
OSD_METHOD = "OSD_CS"     # combination-sweep OSD
OSD_ORDER = 7


def make_bposd_phenom(H_st, n, m, T, p, q):
    """A batched decode_fn: detectors[shots,(T+1)*m] -> net data X-correction Ê[shots,n] (uint8).
    BP-OSD decodes the full space-time correction over [e_0..e_{T-1}|f_0..f_{T-1}]; we return
    the XOR of the decoded per-round data blocks (the net physical correction on the data)."""
    channel = P.channel_prior(n, m, T, p, q)
    dec = BpOsdDecoder(sp.csr_matrix(H_st), channel_probs=list(channel), max_iter=BP_MAX_ITER,
                       bp_method=BP_METHOD, ms_scaling_factor=MS_SCALE,
                       schedule=BP_SCHEDULE, osd_method=OSD_METHOD, osd_order=OSD_ORDER)
    ecols = T * n

    def decode_fn(detectors):
        detectors = np.asarray(detectors, dtype=np.uint8)
        out = np.empty((detectors.shape[0], n), dtype=np.uint8)
        for i in range(detectors.shape[0]):
            corr = dec.decode(detectors[i])
            e_dec = corr[:ecols].reshape(T, n).astype(np.uint8)
            out[i] = np.bitwise_xor.reduce(e_dec, axis=0)     # net data correction
        return out

    return decode_fn


def wilson_hi(fails, n, z=1.96):
    if n == 0:
        return 0.0
    p = fails / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def wilson_lo(fails, n, z=1.96):
    if n == 0:
        return 0.0
    p = fails / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (center - margin) / denom)


def run_code(code: str, shots: int | None = None) -> list[dict]:
    HX, HZ = P.build_bb_code(code)
    prm = P.code_params(HX, HZ)
    cs = P.code_str(code)
    T = P.ROUNDS[code]
    m, n = HZ.shape
    shots = shots or P.DEFAULT_SHOTS
    print(f"\n=== {code} = {cs}  (n={prm['n']}, k={prm['k']}, CSS-commute={prm['css_commute']}) "
          f"— BP-OSD, PHENOMENOLOGICAL T={T} rounds, q=p ===")
    print(f"{'p':>6} {'q':>6} {'T':>3} {'shots':>7} {'fails':>7} {'LER':>12} "
          f"{'95%lo':>11} {'95%hi':>11} {'ms/shot':>9}")
    rows = []
    for p in P.P_SWEEP:
        q = P.q_for(p)
        decode = make_bposd_phenom(P.build_spacetime_check(HZ, T), n, m, T, p, q)
        res = P.evaluate_ler_phenom(decode, code, p, shots)
        lat_us = P.measure_latency_us(decode, code, p)
        hi = wilson_hi(res["failures"], res["shots"])
        lo = wilson_lo(res["failures"], res["shots"])
        print(f"{p:>6.2f} {q:>6.2f} {T:>3} {res['shots']:>7} {res['failures']:>7} "
              f"{res['logical_error_rate']:>12.4e} {lo:>11.4e} {hi:>11.4e} "
              f"{lat_us/1e3:>9.2f}")
        rows.append({"code": code, "code_str": cs, "n": prm["n"], "k": prm["k"],
                     "noise": "phenomenological", "T": T, "p": p, "q": q,
                     "shots": res["shots"], "failures": res["failures"],
                     "ler": res["logical_error_rate"], "ler_95lo": lo, "ler_95hi": hi,
                     "ms_per_shot": lat_us / 1e3, "us_per_round": lat_us / T})
    return rows


def main():
    argv = sys.argv[1:]
    # allow "--shots N" override for a quick pass; default is the frozen DEFAULT_SHOTS
    shots = None
    if "--shots" in argv:
        i = argv.index("--shots"); shots = int(argv[i + 1]); del argv[i:i + 2]
    codes = [a for a in argv if not a.startswith("--")] or ["bb72", "bb144"]

    all_rows = []
    for c in codes:
        all_rows += run_code(c, shots)

    csv_path = os.path.join(RESULT_DIR, "p1_qldpc_phenom.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nsaved {csv_path}")

    # append to results.tsv: primary = phenom LER at reference code (bb72) & p=P_REF
    ref_code = "bb72" if "bb72" in codes else codes[0]
    ref = next(r for r in all_rows if r["code"] == ref_code and abs(r["p"] - P.P_REF) < 1e-9)
    tsv = os.path.join(RESULT_ROOT, "results.tsv")
    with open(tsv, "a") as f:
        f.write(f"bposdph0\t{ref['ler']:.6e}\t0.0\trun\t"
                f"BP-OSD {ref['code_str']} PHENOM T={ref['T']} (p=q={P.P_REF}); "
                f"fails={ref['failures']}/{ref['shots']}\n")
    print(f"baseline reference ({ref_code} {ref['code_str']}, PHENOM T={ref['T']}, p=q={P.P_REF}): "
          f"LER={ref['ler']:.4e}  ({ref['failures']}/{ref['shots']})  "
          f"[multi-round space-time BP-OSD — the harder, realistic bar to beat]")


if __name__ == "__main__":
    main()
