"""
P1 frozen baseline: PyMatching (MWPM / Sparse Blossom) on the surface code across the
pinned distances. This is the reference logical error rate the neural decoder must beat.
(BP-OSD is the qLDPC-family baseline and is added with the color/BB circuits.)

Run (CPU only, no GPU):
    python baseline_mwpm.py
Writes: results/results.tsv rows + results/exp1/p1_mwpm_baseline.csv
"""
from __future__ import annotations
import os, csv
import numpy as np
import pymatching

import prepare as P

FAMILY = "surface"
# Result directory: repo-root results/ by default; override with $P1_RESULT_DIR.
RESULT_ROOT = os.environ.get(
    "P1_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
RESULT_DIR = os.path.join(RESULT_ROOT, "exp1")
os.makedirs(RESULT_DIR, exist_ok=True)


def mwpm_decode_fn(dem):
    m = pymatching.Matching.from_detector_error_model(dem)
    return lambda det: m.decode_batch(det)


def wilson_hi(fails, n, z=1.96):
    if n == 0:
        return 0.0
    p = fails / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center + margin) / denom


def main():
    rows = []
    print(f"{'d':>3} {'shots':>8} {'fails':>7} {'LER':>12} {'LER/round':>12} "
          f"{'95%hi':>12} {'us/round':>10}")
    for d in P.PINNED_DISTANCES[FAMILY]:
        ds = P.generate_stim_dataset(FAMILY, d)
        decode = mwpm_decode_fn(ds["dem"])
        ler = P.evaluate_ler(decode, FAMILY, d)
        lat = P.measure_latency_us(decode, FAMILY, d)
        hi = wilson_hi(ler["failures"], ler["shots"])
        print(f"{d:>3} {ler['shots']:>8} {ler['failures']:>7} "
              f"{ler['logical_error_rate']:>12.3e} {ler['ler_per_round']:>12.3e} "
              f"{hi:>12.3e} {lat:>10.3f}")
        rows.append({"distance": d, "shots": ler["shots"], "failures": ler["failures"],
                     "ler": ler["logical_error_rate"], "ler_per_round": ler["ler_per_round"],
                     "ler_95hi": hi, "us_per_round": lat})

    # save the exact data every figure and table is drawn from
    csv_path = os.path.join(RESULT_DIR, "p1_mwpm_baseline.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # append to results.tsv (primary = LER at the P1 baseline distance d=7)
    tsv = os.path.join(RESULT_ROOT, "results.tsv")
    d7 = next(r for r in rows if r["distance"] == 7)
    with open(tsv, "a") as f:
        f.write(f"mwpm000\t{d7['ler']:.6e}\t0.0\trun\t"
                f"baseline MWPM surface d=7 (p=1e-3,r=25); LER/round={d7['ler_per_round']:.3e}\n")
    print(f"\nsaved {csv_path}")
    print(f"baseline reference (d=7): LER={d7['ler']:.3e}  LER/round={d7['ler_per_round']:.3e}  "
          f"latency={d7['us_per_round']:.2f} us/round  (target for student: <{P.LATENCY_TARGET_US} us/round)")


if __name__ == "__main__":
    main()
