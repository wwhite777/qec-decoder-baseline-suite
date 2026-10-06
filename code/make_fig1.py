"""Figure 1 of the paper: the decoding task for machine-learning readers and the three evaluation settings.
Vector PDF (and a 300-dpi PNG) drawn at the printed width (5.5 in). No data are plotted.
Usage: python code/make_fig1.py  -> figures/fig1_overview.pdf, figures/fig1_overview.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.environ.get("P1_FIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)
# DejaVu Sans ships with matplotlib: the figure renders identically on any machine and embeds as TrueType (pdf.fonttype 42)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.2, "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans"})
INK, GREY = "#222222", "#666666"
CA, CB, CC = "#1f77b4", "#2ca02c", "#d62728"


def box(ax, x, y, w, h, text, fc="#f4f4f4", ec="#888888", fs=7.2, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012", fc=fc, ec=ec, lw=0.7))
    tx = x + (w / 2 if ha == "center" else 0.012)
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK, weight=weight, linespacing=1.25)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=GREY))


def build():
    fig = plt.figure(figsize=(5.5, 3.05))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    FS = 6.0
    # top row: the decoding task as a prediction problem
    ax.text(0.012, 0.965, "The decoding task", fontsize=7.0, weight="bold", color=INK, va="center")
    y, h = 0.70, 0.205
    box(ax, 0.012, y, 0.16, h, "encoded qubits\n+ physical noise\nat rate $p$", fs=FS)
    arrow(ax, 0.177, y + h / 2, 0.202, y + h / 2)
    import numpy as np
    rng = np.random.default_rng(3)
    bits = rng.random((5, 8)) < 0.2
    gx, gy = 0.208, y + 0.045
    for i in range(5):
        for j in range(8):
            ax.add_patch(plt.Rectangle((gx + j * 0.0125, gy + i * 0.028), 0.0108, 0.024,
                                       fc=("#444444" if bits[i, j] else "#e6e6e6"), ec="none"))
    ax.text(gx + 0.05, y + 0.012, "syndrome bits", ha="center", fontsize=FS - 0.4, color=GREY)
    arrow(ax, 0.315, y + h / 2, 0.34, y + h / 2)
    box(ax, 0.345, y, 0.135, h, "decoder\n(classical or\nlearned)", fc="#fff6e6", ec="#d9a441", fs=FS)
    arrow(ax, 0.485, y + h / 2, 0.51, y + h / 2)
    box(ax, 0.515, y, 0.165, h, "predicted logical\nflip(s), compared\nwith the true flip(s)", fs=FS)
    arrow(ax, 0.685, y + h / 2, 0.71, y + h / 2)
    box(ax, 0.715, y, 0.273, h, "failure: any logical bit is wrong\nLER $=f/N$ over $N$ shots, with an\nexact 95% Clopper\u2013Pearson interval", fs=FS)

    # bottom: the three evaluation settings
    ax.text(0.012, 0.6, "Three evaluation settings, each with its own reference decoder (never compared across settings)",
            fontsize=7.0, weight="bold", color=INK, va="center")
    cols = [0.012, 0.058, 0.31, 0.61, 0.785]
    heads = ["", "code", "noise model", "decoder input", "reference decoder"]
    for x, t in zip(cols, heads):
        ax.text(x + 0.004, 0.535, t, fontsize=FS - 0.2, color=GREY, va="center", style="italic")
    rows = [("A", CA, "rotated surface code\n$d=5,7,9,11$", "circuit level: gates, measurements\nand resets can fail",
             "25-round detector\nhistory", "MWPM\n(PyMatching)"),
            ("B", CB, "bivariate-bicycle (BB) codes\n$[[72,12,6]]$, $[[144,12,12]]$", "code capacity: data flips only,\nperfect checks, one round",
             "one syndrome", "BP-OSD (ldpc)"),
            ("C", CC, "BB codes $[[72,12,6]]$ ($T{=}6$)\nand $[[144,12,12]]$ ($T{=}12$)", "phenomenological: data and\nmeasurement flips over $T$ rounds",
             "$T{+}1$ rounds of\ndetection events", "BP-OSD on the\nspace-time checks")]
    yy = 0.43
    for tag, col, code, noise, inp, dec in rows:
        ax.add_patch(FancyBboxPatch((0.012, yy - 0.052), 0.976, 0.104, boxstyle="round,pad=0.004,rounding_size=0.01",
                                    fc="white", ec=col, lw=0.8))
        ax.text(0.024, yy, tag, fontsize=8.5, weight="bold", color=col, va="center")
        for x, t in zip(cols[1:], (code, noise, inp, dec)):
            ax.text(x + 0.004, yy, t, fontsize=FS, color=INK, va="center", linespacing=1.15)
        yy -= 0.122
    ax.text(0.5, 0.03, "Protocol: failure count and exact interval for every LER  \u00b7  paired McNemar test on the same shots  "
            "\u00b7  single-frame call latency", ha="center", fontsize=FS - 0.4, color=GREY, va="bottom")
    return fig


def main():
    fig = build()
    for ext, kw in (("pdf", {"metadata": {"CreationDate": None}}), ("png", {"dpi": 300})):  # no timestamp: reruns are byte-identical
        fig.savefig(os.path.join(OUT_DIR, f"fig1_overview.{ext}"), **kw)
    print("saved", os.path.join(OUT_DIR, "fig1_overview.pdf"))


if __name__ == "__main__":
    main()
