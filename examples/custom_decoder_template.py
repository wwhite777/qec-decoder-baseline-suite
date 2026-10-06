"""Template: run your own decoder through the harness and compare it with the reference.

Copy this file and replace AlwaysNoFlip with your decoder. The only requirement is a method
decode(frame) -> 1-D 0/1 array (optionally also decode_batch(frames) -> 2-D array):
  Arm A (frame = detector bits):           length 1, the predicted flip of observable 0.
  Arm B (frame = syndrome H_Z e):          length n (correction e_hat) or k=12 (L_Z e mod 2).
  Arm C (frame = space-time detector bits): length n (net data correction), T*n+T*m (full
                                            space-time correction) or k=12 (L_Z E mod 2).
Logical predictions use the published basis data/logicals/{code}_LZ.npy (data.LZ; the true
labels of a loaded set are data.labels).

This example decoder always predicts "no logical flip" on Arm B (a length-k zero vector), so
it fails on every shot whose error flips at least one logical. It is a floor, not a decoder.

Usage (CPU):  python examples/custom_decoder_template.py [out_dir]
With out_dir, the two failure vectors are saved in the format of code/compare.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import harness as H  # noqa: E402


class AlwaysNoFlip:
    """Predicts L_Z e = 0 for every syndrome."""

    def __init__(self, k=12):
        self.k = k

    def decode(self, frame):
        return np.zeros(self.k, dtype=np.uint8)


def main():
    data = H.load_frozen("B", code="bb72", p=0.04)          # frames: data.frames, shots: data.shots
    mine = H.score(AlwaysNoFlip(k=data.LZ.shape[0]), data)
    ref = H.score(H.reference_for(data), data)               # BP-OSD, OSD_CS order 7
    print("candidate ", H.fmt_result(mine))
    print("reference ", H.fmt_result(ref))
    print("paired    ", H.fmt_compare(H.compare(mine["fail"], ref["fail"])))
    if len(sys.argv) > 1:
        out = sys.argv[1]
        os.makedirs(out, exist_ok=True)
        H.save_failures(os.path.join(out, "cand.fail.npy"), mine["fail"])
        H.save_failures(os.path.join(out, "ref.fail.npy"), ref["fail"])
        print(f"saved; compare with: python code/compare.py {out}/cand.fail.npy {out}/ref.fail.npy "
              f"--shots {data.shots}")


if __name__ == "__main__":
    main()
