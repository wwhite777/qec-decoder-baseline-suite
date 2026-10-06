"""Collect the outputs of code/v3/driver.py and code/v3/latency_session.py into results/v3/: one CSV of fixed-sample runs, the packed
per-shot failure vectors, the paired tests, the latency sessions (summaries + per-call times) and a SHA256SUMS file.
Usage: python code/v3/export_v3_results.py <runs_dir> <results_v3_dir>
<runs_dir> holds main/ (driver output) and latency/ (session outputs); the paired tests are recomputed from the
exported failure vectors by code/v3/paired_tests.py."""
import csv, glob, hashlib, json, os, shutil, subprocess, sys
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(os.path.join(DST, "failure_vectors"), exist_ok=True)
os.makedirs(os.path.join(DST, "latency_sessions"), exist_ok=True)
rows = []
for f in sorted(glob.glob(os.path.join(SRC, "main", "*.json"))):
    r = json.load(open(f)); c = r["config"]
    rows.append({"id": c["id"], "arm": c["arm"], "code_or_d": c.get("code", f"d{c.get('d', '')}"), "T": c.get("T", 25 if c["arm"] == "A" else ""),
                 "p": c["p"], "decoder": c.get("decoder", "mwpm" if c["arm"] == "A" else "osd7"), "seed": c["seed"],
                 "shots": r["shots"], "failures": r["failures"], "ler": r["ler"], "cp95_lo": r["cp95_lo"], "cp95_hi": r["cp95_hi"],
                 "fail_vector_sha256": r["fail_vector_sha256"], "seconds": round(r["seconds"], 2),
                 "stim": r["versions"]["stim"], "pymatching": r["versions"]["pymatching"], "ldpc": r["versions"]["ldpc"],
                 "numpy": r["versions"]["numpy"], "python": r["versions"]["python"], "cpu": r["cpu"]})
    shutil.copy2(f.replace(".json", ".fail.npy"), os.path.join(DST, "failure_vectors", c["id"] + ".fail.npy"))
with open(os.path.join(DST, "fixed_sample_runs.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "paired_tests.py"), DST],
               check=True, stdout=subprocess.DEVNULL)
lat = []
for f in sorted(glob.glob(os.path.join(SRC, "latency", "s*_L_*.json"))):
    r = json.load(open(f)); c = r["config"]
    lat.append({"session": r["session"], "id": c["id"], "arm": c["arm"], "code_or_d": c.get("code", f"d{c.get('d', '')}"),
                "T": c.get("T", ""), "p": c["p"], "decoder": c.get("decoder", "mwpm" if c["arm"] == "A" else "osd7"),
                "calls": r["n"], "warmup": r["warmup"], "mean_us": r["mean_us"], "p50_us": r["p50_us"], "p90_us": r["p90_us"],
                "p95_us": r["p95_us"], "p99_us": r["p99_us"], "max_us": r["max_us"],
                "p50_ci95_lo": r["p50_ci95_us"][0], "p50_ci95_hi": r["p50_ci95_us"][1],
                "p99_ci95_lo": r["p99_ci95_us"][0], "p99_ci95_hi": r["p99_ci95_us"][1],
                "bp_not_converged": r.get("bp_not_converged", ""), "slowest5_not_converged": r.get("share_of_slowest_5pct_not_converged", ""),
                "core": r["core"], "loadavg_start": r["loadavg_start"][0], "position_in_session": r["position_in_session"],
                "seed": r["seed"], "cpu": r["cpu"]})
    shutil.copy2(f.replace(".json", ".npz"), os.path.join(DST, "latency_sessions", os.path.basename(f).replace(".json", ".npz")))
with open(os.path.join(DST, "latency_sessions.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(lat[0].keys())); w.writeheader(); w.writerows(lat)
lines = []
for root, _, files in os.walk(DST):
    for fn in sorted(files):
        if fn == "SHA256SUMS":
            continue
        p = os.path.join(root, fn)
        lines.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {os.path.relpath(p, DST)}")
open(os.path.join(DST, "SHA256SUMS"), "w").write("\n".join(sorted(lines, key=lambda s: s.split('  ')[1])) + "\n")
print(f"{len(rows)} runs, {len(lat)} latency sessions exported; {len(lines)} files hashed")
