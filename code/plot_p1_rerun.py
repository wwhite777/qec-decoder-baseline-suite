"""
P1 artifact figures — the classical QEC-decoder baseline suite, EXACT data from the
rigor-pass CSVs, with Clopper-Pearson binomial CIs on every logical-error-rate point and
the streaming (online) single-shot latency distribution.

Honest framing (see docs/p1_arms.md, docs/p1_neural_gap.md):
  * THREE separate benchmark arms with DIFFERENT noise models and NON-comparable p axes —
    never plotted on a shared axis. A: surface / MWPM / circuit-level. B: BB / BP-OSD /
    code-capacity. C: BB / BP-OSD / phenomenological (space-time).
  * Every LER carries its exact Clopper-Pearson 95% CI and failure count; under-resolved
    rare-event points (bb144 code-capacity p=0.02 = 2/50k, surface d=11 = 0/100k) are drawn
    as BOUNDS, not values.
  * Panel D: ONLINE per-shot latency (p50/p95/p99), showing the OSD heavy right tail — the
    operational figure a real-time decoder must meet, distinct from batch throughput.
  * The neural decoders are NOT plotted as competitors: they are negative controls (strawmen,
    about 6x to about 100x worse); a competitive learned decoder is scoped future work.
"""
import os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Result directory: repo-root results/ by default; override with $P1_RESULT_DIR.
RESULT_ROOT = os.environ.get(
    "P1_RESULT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
EXP1 = os.path.join(RESULT_ROOT, "exp1")
EXP3 = os.path.join(RESULT_ROOT, "exp3")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures",
                   "fig2_baselines.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cp_yerr(ler, lo, hi):
    """Asymmetric error bars from Clopper-Pearson [lo, hi] around point ler."""
    return np.array([[max(ler - lo, 0.0)], [max(hi - ler, 0.0)]])


def main():
    surf = load(os.path.join(EXP1, "p1_mwpm_baseline_ci.csv"))
    cc = load(os.path.join(EXP3, "p1_qldpc_bposd_ci.csv"))
    ph = load(os.path.join(EXP3, "p1_qldpc_phenom_ci.csv"))
    lat = load(os.path.join(EXP1, "p1_latency.csv"))

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.2))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ---- Panel A: surface / MWPM / circuit-level, LER vs distance -------------
    for r in surf:
        d = int(r["distance"]); fails = int(r["failures"])
        ler = float(r["ler"]); lo = float(r["ler_cp_lo"]); hi = float(r["ler_cp_hi"])
        if fails == 0:  # 0/100k -> upper bound only; draw a downward caret at the CP upper bound
            axA.plot([d], [hi], marker="v", ms=11, color="#d62728", mfc="white", mew=1.6, zorder=5)
            axA.annotate(f"0/{int(r['shots'])//1000}k\n(bound ≤{hi:.1e})", (d, hi),
                         textcoords="offset points", xytext=(6, 4), fontsize=7.5, color="#d62728")
        else:
            axA.errorbar([d], [ler], yerr=cp_yerr(ler, lo, hi), marker="o", ms=8, capsize=5,
                         color="#1f77b4", lw=1.8, zorder=5)
            axA.annotate(f"{fails}/{int(r['shots'])//1000}k", (d, ler),
                         textcoords="offset points", xytext=(6, 5), fontsize=7.5, color="#333")
    axA.set_yscale("log"); axA.set_xticks([5, 7, 9, 11])
    axA.set_xlabel("code distance $d$", fontsize=10)
    axA.set_ylabel("logical error rate per shot (25 rounds)", fontsize=9.5)
    axA.set_title("Arm A — surface / MWPM / circuit-level ($p=10^{-3}$)\n"
                  "sub-threshold suppression; $d{=}11$ is a 0-failure upper bound", fontsize=9.5)
    axA.grid(True, which="both", ls=":", alpha=0.5)

    # ---- Panel B: BB / BP-OSD / code-capacity, LER vs p ----------------------
    codes = {"bb72": ("[[72,12,6]]", "#2ca02c", "o"), "bb144": ("[[144,12,12]]", "#9467bd", "s")}
    for cd, (lab, col, mk) in codes.items():
        rs = [r for r in cc if r["code"] == cd]
        for r in rs:
            p = float(r["p"]); ler = float(r["ler"]); lo = float(r["ler_cp_lo"]); hi = float(r["ler_cp_hi"])
            axB.errorbar([p], [ler], yerr=cp_yerr(ler, lo, hi), marker=mk, ms=7, capsize=4,
                         color=col, lw=1.6, zorder=5)
            if int(r["failures"]) < 30:  # flag under-resolved rare-event point as a bound
                axB.annotate(f"{int(r['failures'])}/{int(r['shots'])//1000}k\nunder-resolved",
                             (p, ler), textcoords="offset points", xytext=(5, -22),
                             fontsize=7, color=col)
        axB.plot([], [], marker=mk, color=col, lw=1.6, label=lab)
    axB.set_yscale("log"); axB.set_xlabel("code-capacity $p$ (i.i.d. data flip)", fontsize=10)
    axB.set_ylabel("logical error rate per shot", fontsize=9.5)
    axB.set_title("Arm B — BB / BP-OSD / code-capacity\n"
                  "distance scaling ($[[144,12,12]]$ below $[[72,12,6]]$ at low $p$)", fontsize=9.5)
    axB.legend(fontsize=9, loc="lower right"); axB.grid(True, which="both", ls=":", alpha=0.5)

    # ---- Panel C: BB / BP-OSD / phenomenological (space-time), LER vs p -------
    rs = [r for r in ph if r["code"] == "bb72"]
    P = [float(r["p"]) for r in rs]; L = [float(r["ler"]) for r in rs]
    ye = np.hstack([cp_yerr(float(r["ler"]), float(r["ler_cp_lo"]), float(r["ler_cp_hi"])) for r in rs])
    axC.errorbar(P, L, yerr=ye, marker="D", ms=7, capsize=4, color="#d62728", lw=1.8, zorder=5)
    for r in rs:
        axC.annotate(f"{int(r['failures'])}/{int(r['shots'])//1000}k",
                     (float(r["p"]), float(r["ler"])), textcoords="offset points",
                     xytext=(6, -12), fontsize=7, color="#333")
    axC.set_xlabel("phenomenological $p=q$ (per-round data & measure flip)", fontsize=10)
    axC.set_ylabel("logical error rate per space-time frame", fontsize=9.5)
    axC.set_ylim(-0.03, 1.05)
    axC.set_title("Arm C — BB / BP-OSD / phenomenological ($[[72,12,6]]$, $T{=}6$)\n"
                  "steep threshold; near-certain failure once $p\\geq0.04$", fontsize=9.5)
    axC.grid(True, ls=":", alpha=0.5)

    # ---- Panel D: ONLINE per-shot latency p50/p95/p99 ------------------------
    order = ["MWPM d=7", "MWPM d=11", "BP-OSD bb72\ncode-cap", "BP-OSD bb144\ncode-cap",
             "BP-OSD bb72\nphenom T=6"]
    p50 = [float(r["p50_us"]) for r in lat]
    p95 = [float(r["p95_us"]) for r in lat]
    p99 = [float(r["p99_us"]) for r in lat]
    x = np.arange(len(order))
    cols = ["#1f77b4", "#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
    axD.bar(x, p50, 0.55, color=cols, edgecolor="black", lw=0.6, alpha=0.85, label="median (p50)")
    # whisker from p50 to p99, tick at p95
    for xi, m, p95i, p99i in zip(x, p50, p95, p99):
        axD.plot([xi, xi], [m, p99i], color="black", lw=1.3, zorder=4)
        axD.plot([xi - 0.14, xi + 0.14], [p95i, p95i], color="black", lw=1.3, zorder=4)
        axD.plot([xi - 0.10, xi + 0.10], [p99i, p99i], color="#d62728", lw=1.8, zorder=4)
        axD.annotate(f"p99 {p99i:.0f}µs", (xi, p99i), textcoords="offset points",
                     xytext=(0, 3), ha="center", fontsize=6.8, color="#d62728")
    axD.set_yscale("log"); axD.set_xticks(x); axD.set_xticklabels(order, fontsize=7.6)
    axD.set_ylabel("online decode latency (µs / call, log)", fontsize=9.5)
    axD.plot([], [], color="black", lw=1.3, label="p95 tick / whisker to p99")
    axD.plot([], [], color="#d62728", lw=1.8, label="p99 (OSD heavy tail)")
    axD.set_title("Online single-shot latency (p50/p95/p99)\n"
                  "OSD fallback gives the phenom decoder a heavy right tail", fontsize=9.5)
    axD.legend(fontsize=7.6, loc="upper left"); axD.grid(True, which="both", axis="y", ls=":", alpha=0.5)

    fig.suptitle("P1 classical QEC-decoder baseline suite: exact LERs with Clopper-Pearson 95% CIs "
                 "+ online latency (three non-comparable arms)", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print("saved", OUT)

    # console echo for the paper (exact numbers)
    print("\nArm A surface/MWPM/circuit-level p=1e-3:")
    for r in surf:
        print(f"  d={r['distance']:>2}  {int(r['failures']):>3}/{r['shots']}  LER={float(r['ler']):.2e}"
              f"  CP95=[{float(r['ler_cp_lo']):.2e},{float(r['ler_cp_hi']):.2e}]")
    print("Arm B BB/BP-OSD/code-capacity:")
    for r in cc:
        print(f"  {r['code']:>5} p={r['p']}  {int(r['failures']):>5}/{r['shots']}  LER={float(r['ler']):.2e}"
              f"  CP95=[{float(r['ler_cp_lo']):.2e},{float(r['ler_cp_hi']):.2e}]")
    print("Arm C BB/BP-OSD/phenom (bb72,T=6):")
    for r in rs:
        print(f"  p=q={r['p']}  {int(r['failures']):>4}/{r['shots']}  LER={float(r['ler']):.3f}"
              f"  CP95=[{float(r['ler_cp_lo']):.3f},{float(r['ler_cp_hi']):.3f}]")
    print("Online latency p50/p95/p99 (us):")
    for r in lat:
        print(f"  {r['decoder']:>17} {r['config']:<26} "
              f"p50={float(r['p50_us']):.1f} p95={float(r['p95_us']):.1f} p99={float(r['p99_us']):.1f}")


if __name__ == "__main__":
    main()
