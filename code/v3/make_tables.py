"""Write every results table of the paper (and of the README) as Markdown, from the repository result files only:
  results/v3/fixed_sample_runs.csv, results/v3/paired_tests.csv, results/v3/latency_sessions.csv,
  results/exp1/p1_mwpm_baseline.csv, results/exp3/p1_qldpc_bposd.csv, results/exp3/p1_qldpc_phenom.csv.
Rounding as in the paper: one half-up rounding of the exact value (2 significant digits; 3 for the frozen sets).
Usage: python code/v3/make_tables.py [out.md]   (default: print to the terminal)"""
import csv, os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from scipy.stats import beta

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1] if len(sys.argv) > 1 else None


def cp(f, n):
    lo = 0.0 if f == 0 else float(beta.ppf(0.025, f, n - f + 1))
    hi = 1.0 if f == n else float(beta.ppf(0.975, f + 1, n - f))
    return lo, hi


def sci(x, sig=2):
    if x == 0:
        return "0"
    d = Decimal(repr(float(x)))
    e = d.adjusted()
    q = d.scaleb(-e).quantize(Decimal(1).scaleb(-(sig - 1)), rounding=ROUND_HALF_UP)
    if q >= 10:
        e += 1
        q = d.scaleb(-e).quantize(Decimal(1).scaleb(-(sig - 1)), rounding=ROUND_HALF_UP)
    if -2 <= e <= -1:
        return str(d.quantize(Decimal(1).scaleb(e - (sig - 1)), rounding=ROUND_HALF_UP))
    return f"{q}e{e}"


def pct2(x):
    """A percentage with two significant digits (one half-up rounding), trailing zeros dropped: 0.02, 0.058, 0.34, 6.7."""
    from decimal import Decimal, ROUND_HALF_UP
    if x == 0:
        return "0"
    d = Decimal(repr(float(x)))
    return format(d.quantize(Decimal(1).scaleb(d.adjusted() - 1), rounding=ROUND_HALF_UP).normalize(), "f")


def rows(p):
    return list(csv.DictReader(open(os.path.join(REPO, p))))


def ler_cells(f, n, sig=2):
    lo, hi = cp(f, n)
    if f == 0:
        return f"<= {sci(hi, sig)}", f"[0, {sci(hi, sig)}]", "no failures"
    return sci(f / n, sig), f"[{sci(lo, sig)}, {sci(hi, sig)}]", ("low count" if f < 30 else "")


out = []
fresh = {r["id"]: r for r in rows("results/v3/fixed_sample_runs.csv")}
# Arm A
out += ["<!-- TABLE armA -->", "| `p` | `d` | failures / shots | LER | 95% interval | flag |", "|---|---|---|---|---|---|"]
for p in ("0.001", "0.002", "0.003", "0.005"):
    for d in (5, 7, 9, 11):
        r = fresh[f"A_d{d}_p{p}"]; f, n = int(r["failures"]), int(r["shots"])
        a, b, c = ler_cells(f, n)
        out.append(f"| {p} | {d} | {f:,} / {n:,} | {a} | {b} | {c} |")
# Arm B
r = fresh["B_bb144_p0.02"]; f, n = int(r["failures"]), int(r["shots"]); a, b, c = ler_cells(f, n)
out += ["", "<!-- TABLE armB -->", f"ARMB {f:,} / {n:,} = {a}, 95% interval {b}"]
# Arm C
out += ["", "<!-- TABLE armC -->", "| code | `T` | `p = q` | failures / shots | LER | 95% interval | flag |", "|---|---|---|---|---|---|---|"]
for code, T, ps in (("bb72", 6, ("0.002", "0.005", "0.01", "0.015", "0.02")), ("bb144", 12, ("0.005", "0.01", "0.02"))):
    for p in ps:
        r = fresh[f"C_{code}_T{T}_p{p}"]; f, n = int(r["failures"]), int(r["shots"]); a, b, c = ler_cells(f, n)
        name = "`[[72,12,6]]`" if code == "bb72" else "`[[144,12,12]]`"
        out.append(f"| {name} | {T} | {p} | {f:,} / {n:,} | {a} | {b} | {c} |")
# Paired
names = {"B_bb72_p0.04": "B: `[[72,12,6]]`, `p = 0.04`", "C_bb72_T6_p0.01": "C: `[[72,12,6]]`, `T = 6`, `p = 0.01`",
         "C_bb72_T6_p0.02": "C: `[[72,12,6]]`, `T = 6`, `p = 0.02`"}
out += ["", "<!-- TABLE paired -->", "| setting | variant | shots | reference fails | variant fails | `b` | `c` | Holm-adjusted p |",
        "|---|---|---|---|---|---|---|---|"]
for t in rows("results/v3/paired_tests.csv"):
    ph = float(t["p_holm"])
    ps = "< 1e-15" if ph < 1e-15 else sci(ph)
    out.append(f"| {names[t['setting']]} | {'OSD-0' if t['candidate'] == 'osd0' else 'BP only'} | {int(t['N']):,} | "
               f"{int(t['f_ref']):,} | {int(t['f_cand']):,} | {int(t['b']):,} | {int(t['c']):,} | {ps} |")
# Latency
lab = {"L_A_d7_p0.001": "A: MWPM, `d = 7`, `p = 1e-3` (25-round frame)", "L_A_d11_p0.001": "A: MWPM, `d = 11`, `p = 1e-3` (25-round frame)",
       "L_B_bb72_p0.02": "B: BP-OSD, `[[72,12,6]]`, `p = 0.02` (syndrome)", "L_B_bb144_p0.02": "B: BP-OSD, `[[144,12,12]]`, `p = 0.02` (syndrome)",
       "L_C_bb72_T6_p0.005": "C: BP-OSD, `[[72,12,6]]`, `p = 0.005` (6-round frame)", "L_C_bb72_T6_p0.02": "C: BP-OSD, `[[72,12,6]]`, `p = 0.02` (6-round frame)",
       "L_C_bb72_T6_p0.02_osd0": "C: BP-OSD-0, `[[72,12,6]]`, `p = 0.02`", "L_C_bb72_T6_p0.02_bp": "C: BP only, `[[72,12,6]]`, `p = 0.02`",
       "L_C_bb144_T12_p0.01": "C: BP-OSD, `[[144,12,12]]`, `p = 0.01` (12-round frame)"}
lat = rows("results/v3/latency_sessions.csv")
out += ["", "<!-- TABLE latency -->", "| arm: decoder, setting (frame) | p50 (us) | range | p99 (us) | range | BP not converged |",
        "|---|---|---|---|---|---|"]
for cid, name in lab.items():
    ss = [r for r in lat if r["id"] == cid]
    assert len(ss) == 5, cid
    p50 = [float(r["p50_us"]) for r in ss]; p99 = [float(r["p99_us"]) for r in ss]
    nc = [r["bp_not_converged"] for r in ss]
    ncs = "" if nc[0] == "" else pct2(sum(int(x) for x in nc) / sum(int(r['calls']) for r in ss) * 100) + "%"
    out.append(f"| {name} | {np.median(p50):.0f} | [{min(p50):.0f}, {max(p50):.0f}] | {np.median(p99):,.0f} | "
               f"[{min(p99):,.0f}, {max(p99):,.0f}] | {ncs} |")
# Frozen sets
out += ["", "<!-- TABLE frozen -->", "| arm | configuration | failures / shots | LER | 95% interval | flag |", "|---|---|---|---|---|---|"]
for r in rows("results/exp1/p1_mwpm_baseline.csv"):
    f, n = int(r["failures"]), int(r["shots"]); a, b, c = ler_cells(f, n, 3)
    out.append(f"| A | surface `d = {r['distance']}`, `p = 1e-3` | {f:,} / {n:,} | {a} | {b} | {c} |")
for r in rows("results/exp3/p1_qldpc_bposd.csv"):
    f, n = int(r["failures"]), int(r["shots"]); a, b, c = ler_cells(f, n, 3)
    out.append(f"| B | `{r['code_str']}`, `p = {float(r['p']):g}` | {f:,} / {n:,} | {a} | {b} | {c} |")
for r in rows("results/exp3/p1_qldpc_phenom.csv"):
    f, n = int(r["failures"]), int(r["shots"]); a, b, c = ler_cells(f, n, 3)
    out.append(f"| C | `[[72,12,6]]`, `T = 6`, `p = {float(r['p']):g}` | {f:,} / {n:,} | {a} | {b} | {c} |")
titles = {"armA": "Arm A: surface code, MWPM, circuit-level noise (fixed-sample runs; paper Table 4)",
          "armB": "Arm B: [[144,12,12]], p = 0.02 (fixed-sample run)",
          "armC": "Arm C: BB codes, BP-OSD, phenomenological noise (fixed-sample runs; paper Table 5)",
          "paired": "Paired comparisons of BP-OSD variants (paper Table 2)",
          "latency": "Single-frame call latency, median and range over five sessions (paper Table 3)",
          "frozen": "Frozen validation sets of the original study (paper Table 6)"}
text = "\n".join(out)
for k, v in titles.items():
    text = text.replace(f"<!-- TABLE {k} -->", f"\n### {v}")
text = text.replace("ARMB ", "") + "\n"
assert "<!--" not in text
if OUT:
    open(OUT, "w").write(text)
    print(f"wrote {OUT}")
else:
    print(text, end="")
