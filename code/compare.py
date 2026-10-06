"""Paired (same-shot) comparison of two decoders from their saved failure vectors.

Usage:
    python code/compare.py <cand.fail.npy> <ref.fail.npy> --shots N

Each file holds np.packbits of a boolean per-shot failure vector over the same N shots in the
same order (harness.save_failures writes this format). Prints b (candidate fails, reference
succeeds), c (candidate succeeds, reference fails), N, delta = (b - c) / N, the exact two-sided
McNemar p-value and the exact Clopper-Pearson 95% interval for b / (b + c); see harness.compare.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("reference")
    ap.add_argument("--shots", type=int, required=True)
    a = ap.parse_args(argv)
    res = H.compare(H.load_failures(a.candidate, a.shots), H.load_failures(a.reference, a.shots))
    ci = res["ci95_b_share"]
    print(f"b      = {res['b']}")
    print(f"c      = {res['c']}")
    print(f"N      = {res['N']}")
    print(f"delta  = {res['delta']:+.6f}   (LER candidate - LER reference on these shots)")
    p = res["p_value"]
    print("McNemar exact two-sided p = " + ("0 (below double-precision range)" if p == 0.0 else f"{p:.6g}"))
    print("b/(b+c) 95% CP = " + ("None (b+c = 0)" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"))


if __name__ == "__main__":
    main()
