"""Mechanism check for the BP-OSD latency tail (exploratory; it does not replace the reported
latency numbers, which come from p1_latency.py).

ldpc's BpOsdDecoder runs ordered-statistics decoding (OSD) only when belief propagation (BP)
does not converge. This script re-times single-shot decodes one call at a time on the same
frames and settings as p1_latency.py and records, per call, whether BP converged. It reports
the fraction of calls in which BP did not converge and how those calls sit in the latency
distribution.

Usage (CPU):  python code/diag_osd_convergence.py [out.json]
"""
import glob, json, os, sys, time
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "frozen_validation")
BP = dict(max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625,
          schedule="serial", osd_method="OSD_CS", osd_order=7)


def one(pattern):
    hits = sorted(glob.glob(os.path.join(DATA, pattern)))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def quantiles(a):
    a = np.asarray(a, dtype=float)
    return {f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 95, 99)} if a.size else {}


def timed(dec, frames, warm=100):
    for i in range(min(warm, len(frames))):
        dec.decode(frames[i])
    lat = np.empty(len(frames))
    conv = np.empty(len(frames), dtype=bool)
    for i in range(len(frames)):
        t0 = time.perf_counter()
        dec.decode(frames[i])
        lat[i] = (time.perf_counter() - t0) * 1e6
        conv[i] = bool(dec.converge)
    return lat, conv


def summary(lat, conv):
    top5 = lat >= np.percentile(lat, 95)
    return {"calls": int(lat.size), "bp_not_converged": int((~conv).sum()),
            "share_of_slowest_5pct_not_converged": float((~conv[top5]).mean()),
            "latency_us_all": quantiles(lat), "latency_us_converged": quantiles(lat[conv]),
            "latency_us_not_converged": quantiles(lat[~conv])}


def main():
    from ldpc import BpOsdDecoder
    sys.path.insert(0, HERE)
    import prepare_qldpc as Q
    import prepare_qldpc_phenom as PH
    out = {}
    for code in ("bb72", "bb144"):
        _, HZ = Q.build_bb_code(code)
        syn = np.load(one(f"qldpc_{code}_p0.02_s50000_seed20260708_*.npz"))["syn"][:3000].astype(np.uint8)
        out[f"code_capacity_{code}_p0.02"] = summary(*timed(BpOsdDecoder(sp.csr_matrix(HZ), error_rate=0.02, **BP), syn))
    _, HZ = PH.build_bb_code("bb72")
    T, (m, n) = 6, HZ.shape
    det = np.load(one("phenom_bb72_T6_p0.02_q0.02_s2100_seed20260709_*.npz"))["det"][:2000].astype(np.uint8)
    dec = BpOsdDecoder(sp.csr_matrix(PH.build_spacetime_check(HZ, T)),
                       channel_probs=list(PH.channel_prior(n, m, T, 0.02, 0.02)), **BP)
    out["phenomenological_bb72_T6_p0.02"] = summary(*timed(dec, det))
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "diag_osd_convergence.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        print(f"{k}: BP did not converge on {v['bp_not_converged']}/{v['calls']} calls; "
              f"share of the slowest 5% that did not converge = {v['share_of_slowest_5pct_not_converged']:.2f}")


if __name__ == "__main__":
    main()
