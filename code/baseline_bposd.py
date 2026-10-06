"""
P1 frozen qLDPC baseline: BP-OSD on the bivariate-bicycle code(s), code-capacity noise.
This is the reference logical error rate the neural decoder must beat on the qLDPC family
(the real battlefield: MWPM does not apply to BB codes).

Decoder: ldpc.BpOsdDecoder (min-sum BP + Ordered-Statistics Decoding fallback), the
Panteleev-Kalachev / Roffe BP-OSD used as the standard BB-code decoder in Bravyi et al.
(Nature 627, 2024). Runs CPU-only (no GPU).

Run (CPU only):
    python baseline_bposd.py            # default: bb72 + bb144
    python baseline_bposd.py bb72       # single code

Writes: results/results.tsv row + results/exp3/p1_qldpc_bposd.csv (exact data).
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import scipy.sparse as sp
from ldpc import BpOsdDecoder

import prepare_qldpc as Q

# Result directory: repo-root results/ by default; override with $P1_RESULT_DIR.
RESULT_ROOT = os.environ.get(
    "P1_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
RESULT_DIR = os.path.join(RESULT_ROOT, "exp3")
os.makedirs(RESULT_DIR, exist_ok=True)

# BP-OSD hyperparameters (frozen for the baseline)
BP_MAX_ITER = 50
BP_METHOD = "minimum_sum"
MS_SCALE = 0.625
BP_SCHEDULE = "serial"
OSD_METHOD = "OSD_CS"     # combination-sweep OSD
OSD_ORDER = 7


def make_bposd(HZ, p):
    """A batched decode_fn: syndromes[shots,m] -> predicted X-errors[shots,n] (uint8)."""
    dec = BpOsdDecoder(sp.csr_matrix(HZ), error_rate=p, max_iter=BP_MAX_ITER,
                       bp_method=BP_METHOD, ms_scaling_factor=MS_SCALE,
                       schedule=BP_SCHEDULE, osd_method=OSD_METHOD, osd_order=OSD_ORDER)

    def decode_fn(syndromes):
        syndromes = np.asarray(syndromes, dtype=np.uint8)
        out = np.empty((syndromes.shape[0], HZ.shape[1]), dtype=np.uint8)
        for i in range(syndromes.shape[0]):
            out[i] = dec.decode(syndromes[i])
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


def run_code(code: str) -> list[dict]:
    HX, HZ = Q.build_bb_code(code)
    prm = Q.code_params(HX, HZ)
    cs = Q.code_str(code)
    print(f"\n=== {code} = {cs}  (n={prm['n']}, k={prm['k']}, "
          f"CSS-commute={prm['css_commute']}) — BP-OSD, code-capacity ===")
    print(f"{'p':>6} {'shots':>8} {'fails':>7} {'LER':>12} {'95%hi':>12} {'us/shot':>10}")
    rows = []
    for p in Q.P_SWEEP:
        decode = make_bposd(HZ, p)
        res = Q.evaluate_ler(decode, code, p)
        lat = Q.measure_latency_us(decode, code, p)
        hi = wilson_hi(res["failures"], res["shots"])
        print(f"{p:>6.2f} {res['shots']:>8} {res['failures']:>7} "
              f"{res['logical_error_rate']:>12.4e} {hi:>12.4e} {lat:>10.2f}")
        rows.append({"code": code, "code_str": cs, "n": prm["n"], "k": prm["k"],
                     "p": p, "shots": res["shots"], "failures": res["failures"],
                     "ler": res["logical_error_rate"], "ler_95hi": hi,
                     "us_per_shot": lat})
    return rows


def main():
    codes = sys.argv[1:] if len(sys.argv) > 1 else ["bb72", "bb144"]
    all_rows = []
    for c in codes:
        all_rows += run_code(c)

    csv_path = os.path.join(RESULT_DIR, "p1_qldpc_bposd.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nsaved {csv_path}")

    # append to results.tsv: primary = LER at reference code (bb72) & p=P_REF
    ref_code = "bb72" if "bb72" in codes else codes[0]
    ref = next(r for r in all_rows if r["code"] == ref_code and abs(r["p"] - Q.P_REF) < 1e-9)
    tsv = os.path.join(RESULT_ROOT, "results.tsv")
    with open(tsv, "a") as f:
        f.write(f"bposd0\t{ref['ler']:.6e}\t0.0\trun\t"
                f"BP-OSD {ref['code_str']} code-capacity (p={Q.P_REF}); "
                f"fails={ref['failures']}/{ref['shots']}\n")
    print(f"baseline reference ({ref_code} {ref['code_str']}, p={Q.P_REF}): "
          f"LER={ref['ler']:.4e}  ({ref['failures']}/{ref['shots']})  "
          f"[qLDPC battlefield: MWPM N/A -> BP-OSD is the bar to beat]")


if __name__ == "__main__":
    main()
