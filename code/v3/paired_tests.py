"""Exact McNemar tests for the paired decoder variants of the fixed-sample runs, Holm-adjusted over the pre-specified
family of six tests ({OSD-0, BP only} vs the OSD_CS order-7 reference, in three settings). The variants of a setting
share one seed, so they decode identical frames; b counts shots where the variant fails and the reference succeeds,
c the reverse, and under equal error rates b ~ Binomial(b + c, 1/2).
Usage: python code/v3/paired_tests.py [results_v3_dir]   -> prints the tests and writes <dir>/paired_tests.csv
"""
import csv, os, sys
import numpy as np
from scipy.stats import binomtest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "v3")
runs = {r["id"]: r for r in csv.DictReader(open(os.path.join(D, "fixed_sample_runs.csv")))}


def vec(cid):
    n = int(runs[cid]["shots"])
    return np.unpackbits(np.load(os.path.join(D, "failure_vectors", cid + ".fail.npy")))[:n].astype(bool)


tests = []
for key in ("B_bb72_p0.04", "C_bb72_T6_p0.01", "C_bb72_T6_p0.02"):
    ref = f"pair_{key}_osd7"
    fr = vec(ref)
    for v in ("osd0", "bp"):
        cand = f"pair_{key}_{v}"
        assert runs[cand]["seed"] == runs[ref]["seed"] and runs[cand]["shots"] == runs[ref]["shots"], cand
        fc = vec(cand)
        b, c, N = int((fc & ~fr).sum()), int((~fc & fr).sum()), int(fr.size)
        p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
        tests.append({"setting": key, "candidate": v, "N": N, "f_ref": int(fr.sum()), "f_cand": int(fc.sum()),
                      "b": b, "c": c, "delta": (b - c) / N, "p": p})
order = sorted(range(len(tests)), key=lambda i: tests[i]["p"])
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p"]))
    tests[i]["p_holm"] = running
with open(os.path.join(D, "paired_tests.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tests[0].keys()))
    w.writeheader()
    w.writerows(tests)
for t in tests:
    print(f"{t['setting']:18s} {t['candidate']:5s} N={t['N']:>7,}  ref {t['f_ref']:>5,}  variant {t['f_cand']:>5,}  "
          f"b={t['b']:>4} c={t['c']:>3}  p={t['p']:.2e}  Holm {t['p_holm']:.2e}")
