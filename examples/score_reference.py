"""Score the paper's reference decoders through the harness on four frozen sets.

Expected failure counts (the frozen CSVs in results/):
    Arm A  surface d=5, p=1e-3, MWPM             78 / 100000
    Arm B  bb72  p=0.04, BP-OSD (OSD_CS order 7)  3969 / 50000
    Arm B  bb144 p=0.02, BP-OSD (OSD_CS order 7)  2 / 50000
    Arm C  bb72 T=6 p=q=0.02, space-time BP-OSD   106 / 2000

Usage (CPU):  python examples/score_reference.py
Exit 0 only if all four counts match.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import harness as H  # noqa: E402

CASES = [
    (dict(arm="A", d=5, p=0.001), 78),
    (dict(arm="B", code="bb72", p=0.04), 3969),
    (dict(arm="B", code="bb144", p=0.02), 2),
    (dict(arm="C", p=0.02), 106),
]


def main():
    bad = 0
    for kw, expected in CASES:
        data = H.load_frozen(**kw)
        res = H.score(H.reference_for(data), data)
        ok = res["failures"] == expected
        bad += not ok
        print(f"Arm {data.arm}  {H.fmt_result(res)}  expected {expected}: {'OK' if ok else 'MISMATCH'}",
              flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
