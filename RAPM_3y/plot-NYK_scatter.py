#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read nyk_players.csv and plot NYK Athletic Score scatter to PDF
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ===== Global font: Arial =====
mpl.rcParams["font.family"] = "Arial"


def plot_nyk_scores(input_csv: str, output_pdf: str) -> str:
    nyk_df = pd.read_csv(input_csv)

    if "Athletic_Score" not in nyk_df.columns:
        raise ValueError("Input CSV must contain column: Athletic_Score")
    if "Player" not in nyk_df.columns:
        raise ValueError("Input CSV must contain column: Player")
    if "Rank" not in nyk_df.columns:
        raise ValueError("Input CSV must contain column: Rank")

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 4))

    # Data
    x = np.arange(len(nyk_df))
    y = nyk_df["Athletic_Score"].values

    # Scatter
    ax.scatter(
        x, y,
        s=30,
        alpha=0.6,
        c="#006BB6",
        edgecolors="black",
        linewidth=1.5
    )

    # Top-3 annotations (by current order in nyk_df)
    top_3_indices = nyk_df.head(3).index

    for idx, (i, row) in enumerate(nyk_df.iterrows()):
        if i in top_3_indices:
            ax.annotate(
                f"Rank {int(row['Rank'])}\n{row['Player']}",
                xy=(idx, row["Athletic_Score"]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=5,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=1.5),
            )

    # Axes labels
    ax.set_xlabel("Players", fontsize=12, fontweight="bold")
    ax.set_ylabel("Athletic Score", fontsize=12, fontweight="bold")

    # Y limits
    y_min = float(y.min()) - 0.05
    y_max = float(y.max()) + 0.1
    ax.set_ylim(y_min, y_max)

    # Grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Mean line
    mean_score = float(y.mean())
    ax.axhline(
        y=mean_score,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Average Score: {mean_score:.4f}"
    )
    ax.legend(loc="lower right", fontsize=10)

    plt.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(output_pdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save PDF
    plt.savefig(output_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    print(f"散点图已保存至: {output_pdf}")
    return output_pdf


def main():
    print("=" * 60)
    print("NYK球队球员散点图绘制")
    print("=" * 60)

    input_csv = "csv_fold/nyk_players 1.csv"      # 或者 "nyk_players.csv"
    output_pdf = "pdfs/nyk_players_scatter.pdf" # 你原来的路径

    plot_nyk_scores(input_csv, output_pdf)

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print("生成文件:")
    print(f"  1. PDF图表: {output_pdf}")


if __name__ == "__main__":
    main()
