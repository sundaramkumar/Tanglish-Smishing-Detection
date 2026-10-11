"""Figure 1: scam recall by model and message group. Reads results/results.json.

Run:  python src/make_figure.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from common import RESULTS


def main():
    R = json.loads((RESULTS / "results.json").read_text())
    serif = [f for f in ("Times New Roman", "Liberation Serif", "DejaVu Serif")
             if any(f == x.name for x in font_manager.fontManager.ttflist)]
    plt.rcParams["font.family"] = serif[0] if serif else "serif"
    plt.rcParams["font.size"] = 11

    models = list(R["exp1"])
    series = [
        ("English UCI spam", [R["exp1"][m]["en_rec"] for m in models]),
        ("Tanglish translation", [R["exp1"][m]["ta_rec"] for m in models]),
        ("Template Tanglish", [R["exp2"][m]["D_template"]["recall"] for m in models]),
        ("Seed Tanglish", [R["exp2"][m]["A_seed"]["recall"] for m in models]),
        ("Call-script Tanglish", [R["exp2"][m]["B_callscript"]["recall"] for m in models]),
    ]
    shades = ["#1a1a1a", "#555555", "#8a8a8a", "#b5b5b5", "#dddddd"]
    hatches = ["", "", "//", "", ".."]
    x, w = np.arange(len(models)), 0.16

    fig, ax = plt.subplots(figsize=(8.6, 4.0), dpi=220)
    for i, (label, vals) in enumerate(series):
        ax.bar(x + (i - 2) * w, [100 * v for v in vals], w, label=label, color=shades[i],
               edgecolor="black", linewidth=0.5, hatch=hatches[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["MNB", "LR", "SVM", "RF", "SVM (char)"])
    ax.set_ylabel("Scam recall (%)")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.3)
    ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.17))
    plt.tight_layout()
    out = RESULTS / "figure1_recall_by_group.png"
    plt.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
