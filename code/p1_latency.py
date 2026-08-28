"""
P1 RIGOR FIX 2 — Streaming single-shot decode latency (ONLINE latency, not batch throughput).

Motivation: the frozen baselines report a per-round wall-clock
figure computed by timing a whole BATCH and dividing by (shots * rounds). That is amortized
THROUGHPUT. A real-time QEC decoder is called once per syndrome frame and must return before
the next round arrives, so the operationally relevant quantity is the ONLINE per-shot latency
DISTRIBUTION — in particular the TAIL (p95/p99), because a decoder that is fast on average but
occasionally stalls will back up the syndrome queue. Batch throughput != online latency, and
the two can differ by orders of magnitude (vectorized batch calls amortize Python/C++ call
overhead and reuse warmed caches that a one-shot-at-a-time call cannot).

This script times each shot's decode as a SEPARATE decoder call (one syndrome in, one
correction out) with a high-resolution clock, and reports the full latency distribution:
    median, mean, p50, p90, p95, p99, p99.9, max.

Decoders measured (both CPU, matching the frozen baselines):
    * MWPM  : pymatching.Matching.decode()      on the surface-code DEM (per detector frame)
    * BP-OSD: ldpc.BpOsdDecoder.decode()          on the BB code-capacity syndrome (per shot)
It also (for context) times BP-OSD on the phenomenological space-time syndrome, since that is
the harder multi-round decode the paper's realistic arm relies on.

We deliberately DO NOT use the frozen measure_latency_us() (which is batch/amortized). We reuse
the frozen datasets + code/DEM construction only.

Run (CPU; no GPU needed):
    python p1_latency.py [n_shots]
Writes: results/exp1/p1_latency.csv  +  results/p1_latency_note.txt
"""
from __future__ import annotations
import os, sys, csv, time
import numpy as np
import scipy.sparse as sp

import pymatching
from ldpc import BpOsdDecoder

import prepare as P                 # surface / Stim / DEM
import prepare_qldpc as Q           # BB code-capacity
import prepare_qldpc_phenom as PH   # BB phenomenological

# Result directory: repo-root results/ by default; override with $P1_RESULT_DIR.
RESULT_ROOT = os.environ.get(
    "P1_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
EXP1 = os.path.join(RESULT_ROOT, "exp1")
os.makedirs(EXP1, exist_ok=True)

# how many individual shots to time per decoder (each is one timed decode call)
N_SHOTS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
WARMUP = 100

# BP-OSD hyperparameters — identical to the frozen baselines (fair timing)
BPOSD_KW = dict(max_iter=50, bp_method="minimum_sum", ms_scaling_factor=0.625,
                schedule="serial", osd_method="OSD_CS", osd_order=7)


def summarize(lat_us: np.ndarray) -> dict:
    lat_us = np.asarray(lat_us, dtype=np.float64)
    return {
        "n": int(lat_us.size),
        "mean_us": float(lat_us.mean()),
        "p50_us": float(np.percentile(lat_us, 50)),
        "median_us": float(np.median(lat_us)),
        "p90_us": float(np.percentile(lat_us, 90)),
        "p95_us": float(np.percentile(lat_us, 95)),
        "p99_us": float(np.percentile(lat_us, 99)),
        "p999_us": float(np.percentile(lat_us, 99.9)),
        "max_us": float(lat_us.max()),
        "min_us": float(lat_us.min()),
    }


def time_stream(decode_one, frames: np.ndarray, warmup: int = WARMUP) -> np.ndarray:
    """Call decode_one(frame) once per row of `frames`, timing each call separately.
    Returns an array of per-shot latencies in microseconds."""
    # warm up (JIT/allocations/caches) OUTSIDE the measured region
    for i in range(min(warmup, frames.shape[0])):
        decode_one(frames[i])
    lat = np.empty(frames.shape[0], dtype=np.float64)
    for i in range(frames.shape[0]):
        f = frames[i]
        t0 = time.perf_counter()
        decode_one(f)
        lat[i] = (time.perf_counter() - t0) * 1e6
    return lat


def run_mwpm_surface(distance: int, n_shots: int):
    ds = P.generate_stim_dataset("surface", distance)
    det = ds["detectors"][:n_shots].astype(np.uint8)
    m = pymatching.Matching.from_detector_error_model(ds["dem"])
    # single-frame decode: pymatching.decode takes one detector vector -> one prediction
    decode_one = lambda f: m.decode(f)
    lat = time_stream(decode_one, det)
    s = summarize(lat)
    s.update(decoder="MWPM (pymatching)", family="surface", noise="circuit-level (Stim)",
             config=f"d={distance},p={P.PHYSICAL_ERROR_RATE},rounds={P.ROUNDS}",
             unit="per detector frame (=1 shot = 25 rounds)")
    return s


def run_bposd_codecap(code: str, p: float, n_shots: int):
    HX, HZ = Q.build_bb_code(code)
    ds = Q.generate_dataset(code, p)
    syn = ds["syndromes"][:n_shots].astype(np.uint8)
    dec = BpOsdDecoder(sp.csr_matrix(HZ), error_rate=p, **BPOSD_KW)
    decode_one = lambda f: dec.decode(f)
    lat = time_stream(decode_one, syn)
    s = summarize(lat)
    s.update(decoder="BP-OSD (ldpc)", family=code, noise="code-capacity",
             config=f"{Q.code_str(code)},p={p}", unit="per shot (single-shot syndrome)")
    return s


def run_bposd_phenom(code: str, p: float, n_shots: int):
    HX, HZ = PH.build_bb_code(code)
    T = PH.ROUNDS[code]
    m, n = HZ.shape
    q = PH.q_for(p)
    H_st = PH.build_spacetime_check(HZ, T)
    ds = PH.generate_dataset(code, p, shots=min(20_000, max(n_shots + WARMUP, 2000)))
    det = ds["detectors"][:n_shots].astype(np.uint8)
    channel = PH.channel_prior(n, m, T, p, q)
    dec = BpOsdDecoder(sp.csr_matrix(H_st), channel_probs=list(channel), **BPOSD_KW)
    decode_one = lambda f: dec.decode(f)
    lat = time_stream(decode_one, det)
    s = summarize(lat)
    s.update(decoder="BP-OSD (ldpc)", family=code, noise=f"phenomenological (T={T})",
             config=f"{PH.code_str(code)},p=q={p},T={T}",
             unit="per shot (T-round space-time frame)")
    return s


def main():
    print(f"Streaming single-shot latency — {N_SHOTS} individually-timed decode calls each\n"
          f"(ONLINE latency; contrast with the batch-amortized us/round in the CSVs)\n")
    rows = []

    # MWPM on the surface code at the P1 reference distance d=7 (and d=11 for scaling context)
    for d in (7, 11):
        print(f"  MWPM surface d={d} ...", flush=True)
        rows.append(run_mwpm_surface(d, N_SHOTS))

    # BP-OSD code-capacity at the reference p=0.02 for both BB codes
    for code in ("bb72", "bb144"):
        print(f"  BP-OSD code-capacity {code} p=0.02 ...", flush=True)
        rows.append(run_bposd_codecap(code, 0.02, N_SHOTS))

    # BP-OSD phenomenological (the realistic multi-round arm) at p=0.02 for bb72
    print(f"  BP-OSD phenomenological bb72 p=0.02 ...", flush=True)
    rows.append(run_bposd_phenom("bb72", 0.02, min(N_SHOTS, 2000)))

    # --- write CSV ---
    cols = ["decoder", "family", "noise", "config", "unit", "n",
            "median_us", "p50_us", "p90_us", "p95_us", "p99_us", "p999_us",
            "mean_us", "min_us", "max_us"]
    csv_path = os.path.join(EXP1, "p1_latency.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    # --- pretty print + note ---
    lines = []
    lines.append("P1 QEC decoder — STREAMING SINGLE-SHOT (online) decode latency\n")
    lines.append("=" * 100 + "\n")
    lines.append(
        "Each number below is the distribution of INDIVIDUAL per-shot decode calls (one\n"
        "syndrome frame in, one correction out), timed one at a time with perf_counter.\n"
        "This is ONLINE latency. It is NOT the same as the batch-amortized us/round reported\n"
        "in the baseline CSVs (those time a whole vectorized batch and divide by shots*rounds).\n"
        "Batch throughput amortizes per-call overhead and reuses warm caches; the online tail\n"
        "(p95/p99) is what a real-time decoder must meet to keep up with the syndrome stream.\n")
    lines.append("=" * 100 + "\n")
    hdr = (f"{'decoder':>16} {'family':>7} {'noise':>22} {'median':>9} {'p50':>9} "
           f"{'p95':>9} {'p99':>9} {'max':>9}   unit\n")
    lines.append(hdr)
    for r in rows:
        lines.append(
            f"{r['decoder']:>16} {r['family']:>7} {r['noise']:>22} "
            f"{r['median_us']:>8.2f}u {r['p50_us']:>8.2f}u {r['p95_us']:>8.2f}u "
            f"{r['p99_us']:>8.2f}u {r['max_us']:>8.2f}u   {r['unit']}\n")
    lines.append("\n(units: u = microseconds per decode call)\n\n")
    lines.append(
        "READING THE NUMBERS:\n"
        "  * MWPM surface is a per-DETECTOR-FRAME decode; the frame already spans all 25 rounds\n"
        "    of the memory experiment, so to compare to a <1 us/round target divide by 25.\n"
        "  * BP-OSD code-capacity is a single-shot syndrome decode (no time dimension).\n"
        "  * BP-OSD phenomenological decodes a whole T-round space-time frame at once, so its\n"
        "    per-round figure is (value / T). Its tail is dominated by the OSD fallback firing\n"
        "    on hard syndromes -> heavy right tail (p99 >> median), the key online-latency risk.\n\n"
        "CAVEATS (be honest):\n"
        "  * All timings are single-threaded Python-level wall-clock on a shared CPU; absolute\n"
        "    values carry interpreter + scheduler overhead and are an UPPER bound on a tuned\n"
        "    C++/FPGA deployment. The DISTRIBUTION SHAPE (median vs tail) is the transferable\n"
        "    finding, not the absolute microseconds.\n"
        "  * pymatching.decode / ldpc.decode still cross the Python<->C++ boundary once per call;\n"
        "    a native streaming implementation would remove that fixed overhead uniformly.\n"
        "  * This measures the classical reference decoders only. A neural decoder's online\n"
        "    latency must be measured the SAME way (one frame at a time, incl. host<->GPU copy),\n"
        "    NOT as a large-batch GPU throughput number, or the comparison is invalid.\n")
    note = "".join(lines)
    note_path = os.path.join(RESULT_ROOT, "p1_latency_note.txt")
    with open(note_path, "w") as f:
        f.write(note)
    print("\n" + note)
    print(f"saved {csv_path}")
    print(f"saved {note_path}")


if __name__ == "__main__":
    main()
