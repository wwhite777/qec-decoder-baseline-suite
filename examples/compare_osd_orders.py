"""BP-OSD order 7 (reference) vs OSD order 0 vs plain BP on the frozen Arm B set bb72, p=0.04.

All three use min-sum BP (scaling 0.625, serial schedule, max_iter 50). Each decoder returns a
physical correction; failure = residual with odd overlap with a row of the published L_Z.
Plain BP's output does not always reproduce the syndrome; such shots are scored by the same
residual rule. Prints each decoder's f/N with its exact 95% interval and the paired McNemar
comparison of osd0 and bp against osd7 on the same 50,000 shots.

Usage (CPU):  python examples/compare_osd_orders.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import harness as H  # noqa: E402


def main():
    data = H.load_frozen("B", code="bb72", p=0.04)
    res = {}
    for osd in ("osd7", "osd0", "bp"):
        res[osd] = H.score(H.ReferenceBPOSD(data.HZ, error_rate=data.p, osd=osd), data)
        print(f"{osd:<5} {H.fmt_result(res[osd])}", flush=True)
    for cand in ("osd0", "bp"):
        print(f"{cand} vs osd7 (paired): {H.fmt_compare(H.compare(res[cand]['fail'], res['osd7']['fail']))}")


if __name__ == "__main__":
    main()
