# log_revenue_triple_linear_regression.py
# 回归：logRevenue ~ year  win_pct  log(win_pct(last_year))
# 输出：Adjusted R^2 (按你给的公式), MSE, AIC, BIC

import numpy as np
import pandas as pd

CSV_PATH = "revenue_regression/merged_team_year_stats_2016_2024.csv"  # 本地运行请改路径

# ---------------------------
# 1) Load + clean
# ---------------------------
df = pd.read_csv(CSV_PATH)

need_cols = ["year", "win_pct", "win_pct(last_year)", "Revenue(M$)","Market Size"]
missing = [c for c in need_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}. Got columns: {list(df.columns)}")

use = df[need_cols].copy()

for c in need_cols:
    use[c] = pd.to_numeric(use[c], errors="coerce")

# log 需要严格正数
use = use.dropna(subset=need_cols)
use = use[(use["Revenue(M$)"] > 0) & (use["win_pct(last_year)"] > 0)].copy()

use["logRevenue"] = np.log(use["Revenue(M$)"])
use["log_win_pct_last_year"] = np.log(use["win_pct(last_year)"])

# ---------------------------
# 2) Fit OLS via least squares
# ---------------------------
y = use["logRevenue"].to_numpy(dtype=float)

X = use[["year", "win_pct", "log_win_pct_last_year","Market Size"]].to_numpy(dtype=float)
n = X.shape[0]
p = X.shape[1]  # predictors count (不含截距)

# 加截距列
X_design = np.column_stack([np.ones(n), X])  # (n, p+1)

# OLS: beta = (X'X)^(-1) X'y
beta, residuals, rank, svals = np.linalg.lstsq(X_design, y, rcond=None)
y_hat = X_design @ beta
resid = y - y_hat

# ---------------------------
# 3) Metrics
# ---------------------------
SSE = float(np.sum(resid ** 2))
y_bar = float(np.mean(y))
SST = float(np.sum((y - y_bar) ** 2))

# 你给的 R^2：1 − [SSE/(n−p−1)] / [SST/(n−1)]  （这就是 Adjusted R^2）
if n - p - 1 <= 0:
    raise ValueError(f"Not enough data: n={n}, p={p} makes (n-p-1) <= 0.")
if SST <= 0:
    raise ValueError("SST <= 0 (y is constant). R^2 undefined.")

MSE = SSE / (n - p - 1)
R2_adj_formula = 1.0 - (SSE / (n - p - 1)) / (SST / (n - 1))

# Gaussian log-likelihood at MLE sigma^2 = SSE/n
sigma2_mle = SSE / n
lnL = -0.5 * n * (np.log(2.0 * np.pi) + np.log(sigma2_mle) + 1.0)

# AIC=2k−2lnL, BIC=k ln n − 2lnL
# 这里按正态线性模型的常用做法：k = (p+1 个回归系数,含截距) + (1 个方差参数 sigma^2) = p+2
k = p + 2
AIC = 2.0 * k - 2.0 * lnL
BIC = k * np.log(n) - 2.0 * lnL

# ---------------------------
# 4) Print results
# ---------------------------
coef_names = ["intercept", "year", "win_pct", "log(win_pct_last_year)","Market Size"]
print(f"Rows used: n={n}, predictors p={p}")
print("\nCoefficients:")
for name, val in zip(coef_names, beta):
    print(f"  {name}: {val}")

print("\nMetrics:")
print(f"  SSE = {SSE}")
print(f"  SST = {SST}")
print(f"  MSE = {MSE}")
print(f"  R^2 (given formula) = {R2_adj_formula}")
print(f"  lnL = {lnL}")
print(f"  k = {k}")
print(f"  AIC = {AIC}")
print(f"  BIC = {BIC}")

# （可选）保存一份结果
out_coef = "revenue_regression/log_revenue_regression_coefs.csv"
pd.DataFrame({"term": coef_names, "coef": beta}).to_csv(out_coef, index=False)
print("\nSaved coefficients to:", out_coef)
