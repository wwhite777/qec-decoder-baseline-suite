"""
Figure 2 of the paper: the classical QEC-decoder baseline suite, drawn from the exact result CSVs.

  * Three separate benchmark arms with different noise models and non-comparable p axes, never
    plotted on a shared axis. A: surface / MWPM / circuit-level. B: BB / BP-OSD / code capacity.
    C: BB / BP-OSD / phenomenological (space-time), [[72,12,6]] only.
  * Every classical point shows its exact Clopper-Pearson 95% interval and failure count. Points
    with fewer than 30 failures are under-resolved: they are drawn as an open caret at the 95%
    upper limit (the bound the paper reports), with the interval as a thin line.
  * Panel D: online single-shot latency (p50 bar, p95 tick, whisker to p99), the measured
    distribution for this Python implementation on one shared CPU.
Drawn at the printed width (5.5 in) so the text keeps its size in the paper; writes
figures/fig2_baselines.pdf (vector, used by the paper) and figures/fig2_baselines.png (300 dpi).
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
OUT_DIR = os.environ.get(
    "P1_FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
UNDER = 30          # reading rule: fewer than 30 failures -> under-resolved, report the upper limit
FS, FST, FSA = 7.0, 7.4, 6.0   # label, title and annotation font sizes (points, at printed size)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def draw_point(ax, x, r, color, marker, ms=4.5):
    """Resolved point: marker at f/N with its CP interval. Under-resolved: thin interval line and
    an open caret at the CP upper limit. Returns the y used for the annotation."""
    f = int(r["failures"]); ler = float(r["ler"])
    lo, hi = float(r["ler_cp_lo"]), float(r["ler_cp_hi"])
    if f < UNDER:
        if f > 0:
            ax.plot([x, x], [lo, hi], color=color, lw=0.8, alpha=0.6, zorder=3)
        ax.plot([x], [hi], marker="v", ms=ms + 1.5, color=color, mfc="white", mew=1.1, ls="none", zorder=5)
        return hi
    ax.errorbar([x], [ler], yerr=[[ler - lo], [hi - ler]], marker=marker, ms=ms, capsize=2.5,
                color=color, lw=1.0, ls="none", zorder=5)
    return ler


def main():
    surf = load(os.path.join(EXP1, "p1_mwpm_baseline_ci.csv"))
    cc = load(os.path.join(EXP3, "p1_qldpc_bposd_ci.csv"))
    ph = [r for r in load(os.path.join(EXP3, "p1_qldpc_phenom_ci.csv")) if r["code"] == "bb72"]
    lat = load(os.path.join(EXP1, "p1_latency.csv"))
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": FS, "axes.titlesize": FST, "axes.labelsize": FS,
                         "xtick.labelsize": FS - 0.6, "ytick.labelsize": FS - 0.6, "legend.fontsize": FSA + 0.4})
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.75))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # A: surface / MWPM / circuit-level
    for r in surf:
        d = int(r["distance"])
        y = draw_point(axA, d, r, "#1f77b4", "o")
        axA.annotate(f"{int(r['failures'])}/{int(r['shots']) // 1000}k", (d, y), textcoords="offset points",
                     xytext=(4, 2), fontsize=FSA, color="#333")
    axA.set_yscale("log"); axA.set_xticks([5, 7, 9, 11]); axA.set_xlim(4.2, 12.3)
    axA.set_xlabel("code distance $d$"); axA.set_ylabel("LER per 25-round shot")
    axA.set_title("A  surface, MWPM, circuit-level, $p=10^{-3}$", loc="left")
    axA.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    # B: BB / BP-OSD / code capacity
    codes = {"bb72": ("$[[72,12,6]]$", "#2ca02c", "o"), "bb144": ("$[[144,12,12]]$", "#9467bd", "s")}
    for cd, (lab, col, mk) in codes.items():
        for r in [r for r in cc if r["code"] == cd]:
            p = float(r["p"])
            y = draw_point(axB, p, r, col, mk)
            if int(r["failures"]) < UNDER:
                axB.annotate(f"{int(r['failures'])}/{int(r['shots']) // 1000}k", (p, y), textcoords="offset points",
                             xytext=(5, -2), fontsize=FSA, color=col)
        axB.plot([], [], marker=mk, color=col, ls="none", ms=4.5, label=lab)
    axB.set_yscale("log"); axB.set_xticks([0.02, 0.04, 0.06, 0.08]); axB.set_xlim(0.012, 0.088)
    axB.set_xlabel("code-capacity $p$"); axB.set_ylabel("LER per shot")
    axB.set_title("B  BB codes, BP-OSD, code capacity", loc="left")
    axB.legend(loc="lower right", frameon=True, handletextpad=0.2, borderpad=0.3)
    axB.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)

    # C: BB / BP-OSD / phenomenological, [[72,12,6]] only
    P = [float(r["p"]) for r in ph]; L = [float(r["ler"]) for r in ph]
    axC.plot(P, L, color="#d62728", lw=0.8, alpha=0.5, zorder=2)
    for r in ph:
        y = draw_point(axC, float(r["p"]), r, "#d62728", "D", ms=4.0)
        off = {0.02: (5, 2), 0.04: (5, -9), 0.06: (5, -9), 0.08: (2, -12)}[float(r["p"])]
        axC.annotate(f"{int(r['failures'])}/{int(r['shots']) // 1000}k", (float(r["p"]), y), textcoords="offset points",
                     xytext=off, fontsize=FSA, color="#333")
    axC.set_xticks([0.02, 0.04, 0.06, 0.08]); axC.set_xlim(0.012, 0.095); axC.set_ylim(-0.03, 1.08)
    axC.set_xlabel("phenomenological $p=q$ (per round)"); axC.set_ylabel("LER per space-time frame")
    axC.set_title("C  $[[72,12,6]]$, BP-OSD, phenomenological, $T=6$", loc="left")
    axC.grid(True, ls=":", lw=0.4, alpha=0.6)

    # D: online single-shot latency
    names = ["MWPM\n$d$=7", "MWPM\n$d$=11", "BP-OSD\n72 cc", "BP-OSD\n144 cc", "BP-OSD\n72 ph"]
    cols = ["#1f77b4", "#1f77b4", "#2ca02c", "#9467bd", "#d62728"]
    p50 = [float(r["p50_us"]) for r in lat]; p95 = [float(r["p95_us"]) for r in lat]; p99 = [float(r["p99_us"]) for r in lat]
    x = np.arange(len(lat))
    axD.bar(x, p50, 0.55, color=cols, edgecolor="black", lw=0.4, alpha=0.85)
    for xi, m, a, b in zip(x, p50, p95, p99):
        axD.plot([xi, xi], [m, b], color="black", lw=0.8, zorder=4)
        axD.plot([xi - 0.14, xi + 0.14], [a, a], color="black", lw=0.8, zorder=4)
        axD.plot([xi - 0.10, xi + 0.10], [b, b], color="#d62728", lw=1.2, zorder=4)
        axD.annotate(f"{b:.0f}", (xi, b), textcoords="offset points", xytext=(0, 2), ha="center",
                     fontsize=FSA, color="#d62728")
    axD.set_yscale("log"); axD.set_ylim(5, 9000); axD.set_xticks(x); axD.set_xticklabels(names, fontsize=FSA)
    axD.set_ylabel("online latency ($\\mu$s per call)")
    axD.plot([], [], color="black", lw=0.8, label="p95 tick, whisker to p99")
    axD.plot([], [], color="#d62728", lw=1.2, label="p99 (value in $\\mu$s)")
    axD.set_title("D  online single-shot latency", loc="left")
    axD.legend(loc="upper left", frameon=True, handlelength=1.2, borderpad=0.3)
    axD.grid(True, which="both", axis="y", ls=":", lw=0.4, alpha=0.6)

    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        path = os.path.join(OUT_DIR, f"fig2_baselines.{ext}")
        fig.savefig(path, **kw)
        print("saved", path)

    # console echo of the plotted values
    for r in surf:
        print(f"A d={r['distance']:>2} {int(r['failures']):>3}/{r['shots']} LER={float(r['ler']):.3e} "
              f"CP95=[{float(r['ler_cp_lo']):.3e},{float(r['ler_cp_hi']):.3e}]")
    for r in cc:
        print(f"B {r['code']:>5} p={r['p']} {int(r['failures']):>5}/{r['shots']} LER={float(r['ler']):.3e} "
              f"CP95=[{float(r['ler_cp_lo']):.3e},{float(r['ler_cp_hi']):.3e}]")
    for r in ph:
        print(f"C p=q={r['p']} {int(r['failures']):>4}/{r['shots']} LER={float(r['ler']):.4f} "
              f"CP95=[{float(r['ler_cp_lo']):.4f},{float(r['ler_cp_hi']):.4f}]")
    for r in lat:
        print(f"D {r['decoder']:>17} {r['config']:<26} p50={float(r['p50_us']):.1f} p95={float(r['p95_us']):.1f} "
              f"p99={float(r['p99_us']):.1f}")


if __name__ == "__main__":
    main()
