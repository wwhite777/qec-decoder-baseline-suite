"""Figure 2 of the paper, drawn at the printed width (5.5 in) from the result files only:
  results/v3/fixed_sample_runs.csv    fixed-sample runs (filled markers)
  results/exp1, results/exp3          the original frozen-set results (open markers)
  results/v3/latency_sessions.csv     single-frame latency sessions (panel D)
Every point carries its exact two-sided 95% Clopper-Pearson interval; a zero-failure point is a downward triangle at the
upper end of its interval. Writes figures/fig2_baselines.pdf (vector) and .png (300 dpi).
Usage: python code/plot_fig2.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("P1_RESULT_DIR", os.path.join(HERE, "..", "results"))
OUT = os.environ.get("P1_FIG_DIR", os.path.join(HERE, "..", "figures"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Nimbus Sans", "DejaVu Sans"], "mathtext.fontset": "dejavusans",
                     "font.size": 7.0, "axes.titlesize": 7.4, "axes.labelsize": 7.0, "xtick.labelsize": 6.4,
                     "ytick.labelsize": 6.4, "legend.fontsize": 6.0})
DCOL = {5: "#1f77b4", 7: "#ff7f0e", 9: "#2ca02c", 11: "#9467bd"}
CCOL = {"bb72": "#d62728", "bb144": "#8c564b"}


def cp(f, n):
    lo = 0.0 if f == 0 else beta.ppf(0.025, f, n - f + 1)
    hi = 1.0 if f == n else beta.ppf(0.975, f + 1, n - f)
    return lo, hi


def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def point(ax, x, f, n, color, marker, filled=True, ms=3.6):
    lo, hi = cp(f, n)
    if f == 0:
        ax.plot([x], [hi], marker="v", ms=ms + 1, color=color, mfc="white" if not filled else color, ls="none", zorder=5)
        return hi
    y = f / n
    ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], marker=marker, ms=ms, capsize=1.6, lw=0.8, color=color,
                mfc=color if filled else "white", ls="none", zorder=5)
    return y


def main():
    fresh = rows(os.path.join(RES, "v3", "fixed_sample_runs.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(5.5, 4.6))
    axA, axB, axC, axD = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    # A: surface code, LER vs p per distance (fresh) + frozen p=1e-3 (open, offset)
    for d in (5, 7, 9, 11):
        rs = sorted([r for r in fresh if r["arm"] == "A" and r["code_or_d"] == f"d{d}"], key=lambda r: float(r["p"]))
        xs, ys = [], []
        for r in rs:
            xs.append(float(r["p"])); ys.append(point(axA, float(r["p"]), int(r["failures"]), int(r["shots"]), DCOL[d], "o"))
        axA.plot(xs, ys, color=DCOL[d], lw=0.7, alpha=0.6, zorder=2, label=f"$d={d}$")
    for r in rows(os.path.join(RES, "exp1", "p1_mwpm_baseline.csv")):
        d = int(r["distance"])
        point(axA, 1e-3 * 0.93, int(r["failures"]), int(r["shots"]), DCOL[d], "o", filled=False, ms=3.2)
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xticks([1e-3, 2e-3, 3e-3, 5e-3]); axA.set_xticklabels(["0.001", "0.002", "0.003", "0.005"])
    axA.minorticks_off(); axA.set_xlim(8.6e-4, 5.8e-3)
    axA.set_xlabel("physical error rate $p$"); axA.set_ylabel("LER per 25-round shot")
    axA.set_title("A  surface code, MWPM, circuit level", loc="left")
    axA.legend(loc="lower right", ncol=2, frameon=True, handlelength=1.2, columnspacing=0.8, borderpad=0.3)
    axA.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    # B: BB code capacity, frozen (open) + fresh bb144 p=0.02 (filled)
    for r in rows(os.path.join(RES, "exp3", "p1_qldpc_bposd.csv")):
        mk = "o" if r["code"] == "bb72" else "s"
        point(axB, float(r["p"]), int(r["failures"]), int(r["shots"]), CCOL[r["code"]], mk, filled=False)
    for r in fresh:
        if r["arm"] == "B" and r["decoder"] == "osd7" and r["code_or_d"] == "bb144":
            point(axB, float(r["p"]) * 1.04, int(r["failures"]), int(r["shots"]), CCOL["bb144"], "s", filled=True)
    axB.plot([], [], marker="o", color=CCOL["bb72"], mfc="white", ls="none", label="$[[72,12,6]]$")
    axB.plot([], [], marker="s", color=CCOL["bb144"], mfc="white", ls="none", label="$[[144,12,12]]$")
    axB.set_yscale("log"); axB.set_xticks([0.02, 0.04, 0.06, 0.08]); axB.set_xlim(0.014, 0.086)
    axB.set_xlabel("code-capacity $p$"); axB.set_ylabel("LER per shot")
    axB.set_title("B  BB codes, BP-OSD, code capacity", loc="left")
    axB.legend(loc="lower right", frameon=True, handletextpad=0.2, borderpad=0.3)
    axB.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    # C: phenomenological, fresh (filled) both codes + frozen bb72 (open)
    for code, mk in (("bb72", "o"), ("bb144", "s")):
        rs = sorted([r for r in fresh if r["arm"] == "C" and r["decoder"] == "osd7" and r["code_or_d"] == code
                     and not r["id"].startswith("pair_")], key=lambda r: float(r["p"]))
        xs, ys = [], []
        for r in rs:
            y = point(axC, float(r["p"]), int(r["failures"]), int(r["shots"]), CCOL[code], mk)
            if int(r["failures"]) > 0:          # lines join estimates only, never a zero-failure upper limit
                xs.append(float(r["p"])); ys.append(y)
        axC.plot(xs, ys, color=CCOL[code], lw=0.7, alpha=0.6, zorder=2)
    for r in rows(os.path.join(RES, "exp3", "p1_qldpc_phenom.csv")):
        point(axC, float(r["p"]) * 1.06, int(r["failures"]), int(r["shots"]), CCOL["bb72"], "o", filled=False, ms=3.2)
    axC.plot([], [], marker="o", color=CCOL["bb72"], ls="none", label="$[[72,12,6]]$, $T{=}6$")
    axC.plot([], [], marker="s", color=CCOL["bb144"], ls="none", label="$[[144,12,12]]$, $T{=}12$")
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_xticks([0.002, 0.005, 0.01, 0.02, 0.04, 0.08]); axC.set_xticklabels(["0.002", "0.005", "0.01", "0.02", "0.04", "0.08"])
    axC.minorticks_off(); axC.set_xlim(1.7e-3, 0.095)
    axC.set_xlabel("phenomenological $p=q$ (per round)"); axC.set_ylabel("LER per space-time frame")
    axC.set_title("C  BB codes, BP-OSD, phenomenological", loc="left")
    axC.legend(loc="lower right", frameon=True, handletextpad=0.2, borderpad=0.3)
    axC.grid(True, which="both", ls=":", lw=0.4, alpha=0.6)
    # D: latency sessions
    lat = rows(os.path.join(RES, "v3", "latency_sessions.csv"))
    order = [("L_A_d7_p0.001", "A\nd=7"), ("L_A_d11_p0.001", "A\nd=11"), ("L_B_bb72_p0.02", "B\n72"),
             ("L_B_bb144_p0.02", "B\n144"), ("L_C_bb72_T6_p0.005", "C 72\n.005"), ("L_C_bb72_T6_p0.02", "C 72\n.02"),
             ("L_C_bb72_T6_p0.02_osd0", "C 72\nOSD-0"), ("L_C_bb72_T6_p0.02_bp", "C 72\nBP"), ("L_C_bb144_T12_p0.01", "C 144\n.01")]
    cols = ["#1f77b4", "#1f77b4", "#2ca02c", "#2ca02c", "#d62728", "#d62728", "#e377c2", "#7f7f7f", "#8c564b"]
    for i, (cid, _) in enumerate(order):
        ss = [r for r in lat if r["id"] == cid]
        if not ss:
            continue
        p50 = np.median([float(r["p50_us"]) for r in ss]); p99s = [float(r["p99_us"]) for r in ss]
        axD.bar(i, p50, 0.6, color=cols[i], alpha=0.85, edgecolor="black", lw=0.3)
        axD.errorbar([i], [np.median(p99s)], yerr=[[np.median(p99s) - min(p99s)], [max(p99s) - np.median(p99s)]],
                     marker="_", ms=6, mew=1.2, color="black", capsize=1.5, lw=0.7, ls="none")
    axD.set_yscale("log"); axD.set_xticks(range(len(order))); axD.set_xticklabels([o[1] for o in order], fontsize=5.6)
    axD.set_ylabel("single-frame call latency ($\\mu$s)")
    axD.bar([], [], color="#bbbbbb", label="p50 (median of 5 sessions)")
    axD.errorbar([], [], yerr=[], marker="_", color="black", ls="none", label="p99 (median, range)")
    axD.legend(loc="upper left", frameon=True, borderpad=0.3)
    axD.set_title("D  single-frame call latency", loc="left")
    axD.grid(True, which="both", axis="y", ls=":", lw=0.4, alpha=0.6)
    fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 300})):  # no timestamp: reruns are byte-identical
        fig.savefig(os.path.join(OUT, f"fig2_baselines.{ext}"), **kw)
    print("saved", os.path.join(OUT, "fig2_baselines.pdf"))


if __name__ == "__main__":
    main()
