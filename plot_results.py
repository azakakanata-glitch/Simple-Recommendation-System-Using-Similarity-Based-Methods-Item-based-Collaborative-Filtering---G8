"""
plot_results.py
Purpose: draw the Chapter 5 charts from the CSV files that evaluate.py wrote to results/.
How to run: python evaluate.py, then python plot_results.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from src import config

BLUE, GRAY, INK, MUTED = "#2a78d6", "#b9b7b0", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#c9c8c3", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#ecebe7", "axes.axisbelow": True,
})
fmt = lambda v, d=3: f"{v:.{d}f}"
OUT = config.RESULTS_DIR

LABELS = {"B1": "B1 Global\nmean", "B2": "B2 User\nmean", "B3": "B3 Most\npopular",
          "E1": "E1 Cosine\nfirst version", "E2": "E2 Cosine", "E3": "E3 Pearson", "E4": "E4 Jaccard",
          "E5": "E5 User-based\nCosine"}


def bars(df, col, ylabel, fname, highlight="E2", lower_better=True):
    df = df.dropna(subset=[col])
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=200)
    colors = [BLUE if m == highlight else GRAY for m in df["id"]]
    ax.bar([LABELS[m] for m in df["id"]], df[col], color=colors, width=0.66)
    for i, v in enumerate(df[col]):
        ax.annotate(fmt(v, 4), (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    if lower_better:
        ax.set_ylim(0.8, df[col].max() * 1.04)
    else:
        ax.set_ylim(0, df[col].max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v, 2)))
    ax.tick_params(axis="x", labelsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


def sweep(fname_csv, key, xlabel, fname):
    df = pd.read_csv(OUT / fname_csv)
    x = [str(v) if str(v) not in ("nan", "all") else "all" for v in df[key]]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), dpi=200)
    for ax, (col, title, d) in zip(axes, [("rmse", "RMSE", 3), ("coverage", "Coverage", 2),
                                          ("precision", f"Precision@{config.EVAL_K}", 3)]):
        ax.plot(x, df[col], color=BLUE, lw=2, marker="o", ms=5)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel(xlabel)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, d=d: fmt(v, d)))
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / fname)
    plt.close(fig)


if __name__ == "__main__":
    main = pd.read_csv(OUT / "results.csv")
    bars(main[main["id"] != "B3"], "rmse", "RMSE (lower is better)", "fig5_1_rmse.png")
    bars(main[main["id"] != "B1"], "precision", f"Precision@{config.EVAL_K} (higher is better)",
         "fig5_2_precision.png", lower_better=False)
    sweep("sensitivity_k.csv", "k", "number of neighbours k", "fig5_3_sensitivity_k.png")
    sweep("sensitivity_beta.csv", "beta", "shrinkage threshold β", "fig5_4_sensitivity_beta.png")
    print(f"Charts saved to {OUT}")
