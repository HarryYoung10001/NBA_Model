import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ===== Global font: Arial =====
mpl.rcParams["font.family"] = "Arial"

# ---- Config ----
CSV_PATH = Path("player_scores.csv")      # input CSV
SCORE_COL = "Athletic_Score"              # score column name
OUTPUT_PDF = Path("pdfs/ecdf_plot.pdf")        # output PDF

# ---- Style knobs (interfaces) ----
AXIS_LABEL_FONTSIZE = 14   # x/y axis name font size
TICK_LABEL_FONTSIZE = 12   # axis tick value font size

BOLD_AXIS_LABELS = True    # whether x/y axis names are bold
BOLD_TICKS = False         # whether tick labels are bold
BOLD_TITLE = True          # whether title is bold
BOLD_PERCENTILE_LABELS = False  # whether P10/P25... text is bold

# Optional: line width knobs
ECDF_LINEWIDTH = 1.5
PERCENTILE_LINEWIDTH = 1.0

# ---- Load ----
df = pd.read_csv(CSV_PATH)

# If SCORE_COL is absent, fall back to the first numeric column
if SCORE_COL not in df.columns:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("No numeric columns found. Please set SCORE_COL to a numeric column name.")
    SCORE_COL = num_cols[0]

x = df[SCORE_COL].dropna().to_numpy()
x = x[np.isfinite(x)]

# ---- ECDF ----
x_sorted = np.sort(x)
n = len(x_sorted)
y = np.arange(1, n + 1) / n

# Reference percentiles
p10, p25, p50, p75, p90 = np.percentile(x_sorted, [10, 25, 50, 75, 90])

fig, ax = plt.subplots(figsize=(4, 4))

ax.step(x_sorted, y, where="post", linewidth=ECDF_LINEWIDTH)

# ---- Labels with size + bold interface ----
axis_label_weight = "bold" if BOLD_AXIS_LABELS else "normal"
tick_label_weight = "bold" if BOLD_TICKS else "normal"
title_weight = "bold" if BOLD_TITLE else "normal"
pct_label_weight = "bold" if BOLD_PERCENTILE_LABELS else "normal"

ax.set_xlabel(SCORE_COL, fontsize=AXIS_LABEL_FONTSIZE, fontweight=axis_label_weight)
ax.set_ylabel("ECDF", fontsize=AXIS_LABEL_FONTSIZE, fontweight=axis_label_weight)

# Tick label font size + bold
ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_fontweight(tick_label_weight)

# Percentile markers (no explicit colors)
for v, lab in [(p10, "P10"), (p25, "P25"), (p50, "P50"), (p75, "P75"), (p90, "P90")]:
    ax.axvline(v, linewidth=PERCENTILE_LINEWIDTH)
    ax.text(v, 0.02, lab, rotation=90, va="bottom", ha="right",
            fontsize=TICK_LABEL_FONTSIZE, fontweight=pct_label_weight)

ax.set_ylim(0, 1.02)
ax.grid(True, linewidth=0.5, alpha=0.5)

fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)  # PDF output
plt.show()
