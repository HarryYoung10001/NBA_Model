#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为各年龄组创建箱线图，并对平均值进行二次回归分析
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# 读取提取后的数据
df = pd.read_csv('csv_fold/AGE_USG%.csv')

# 获取唯一年龄并排序
ages = sorted(df['AGE'].unique())

# 准备箱线图数据
box_data = []
age_labels = []
means = []  # 平均值
age_means = []  # 对应的年龄

for age in ages:
    age_performance = df[df['AGE'] == age]['Performance'].values
    if len(age_performance) > 0:
        box_data.append(age_performance)
        age_labels.append(int(age))
        means.append(np.mean(age_performance))
        age_means.append(age)

# 创建图形
fig, ax = plt.subplots(figsize=(16, 10))

# 创建箱线图
bp = ax.boxplot(box_data, 
                tick_labels=age_labels,
                patch_artist=True,
                widths=0.6,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='darkred'),
                medianprops=dict(color='blue', linewidth=2),
                boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1.5, alpha=0.7),
                whiskerprops=dict(color='black', linewidth=1.5),
                capwidths=0.5)

# 获取箱线图的位置（1,2,3,...）
positions = np.arange(1, len(ages) + 1)

# 对平均值进行二次回归
z = np.polyfit(age_means, means, 2)
p = np.poly1d(z)

# 创建平滑的回归曲线
x_smooth_age = np.linspace(min(age_means), max(age_means), 200)
y_smooth = p(x_smooth_age)

# 将年龄值转换为箱线图位置
# 使用插值将年龄映射到对应的箱线图位置
x_smooth_positions = np.interp(x_smooth_age, age_means, positions)

# 绘制平均值回归曲线（使用转换后的位置）
ax.plot(x_smooth_positions, y_smooth, 'orange', linewidth=4, 
        label=f'Quadratic Regression on Means\n\(y = {z[0]:.4f}x^2 + {z[1]:.4f}x + {z[2]:.4f}\\)',
        zorder=10)

# 绘制平均值点
ax.scatter(positions, means, 
          color='red', s=100, zorder=13, alpha=1.0,
          label='Mean Values', marker='D', edgecolors='darkred', linewidth=2)

# 从回归曲线找出峰值
peak_age_idx = np.argmax(y_smooth)
peak_age = x_smooth_age[peak_age_idx]
peak_position = x_smooth_positions[peak_age_idx]
peak_performance = y_smooth[peak_age_idx]

# 在峰值处添加垂直线
ax.axvline(x=peak_position, 
          color='green', linestyle='--', linewidth=3, alpha=0.8,
          label=f'Peak Age (from means): {peak_age:.1f} years')

# 计算回归的R²值
y_pred = p(age_means)
ss_res = np.sum((means - y_pred) ** 2)
ss_tot = np.sum((means - np.mean(means)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# 设置标签和标题
ax.set_xlabel('Age (Years)', fontsize=16, fontweight='bold')
ax.set_ylabel('USG%', fontsize=16, fontweight='bold')
ax.set_title('Performance Distribution by Age with Quadratic Regression on MEANS', 
            fontsize=18, fontweight='bold', pad=20)

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# 更新图例元素
legend_elements = [
    plt.Line2D([0], [0], color='blue', linewidth=2, label='Median (Box Plot)'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='red', 
               markeredgecolor='darkred', markersize=10, label='Mean (Box Plot)'),
    plt.Line2D([0], [0], color='orange', linewidth=4, 
               label=f'Quadratic Fit on Means (R²={r_squared:.3f})'),
    plt.Line2D([0], [0], color='green', linestyle='--', linewidth=3,
               label=f'Peak from Means: {peak_age:.1f} yrs')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True, 
         shadow=True, fancybox=True)

# 添加统计信息文本框
textstr = f'Total Players: {len(df)}\nAge Groups: {len(ages)}\n\nRegression on MEANS:\n\\(y = {z[0]:.4f}x^2 + {z[1]:.4f}x + {z[2]:.4f}\\)\n\nR² = {r_squared:.4f}'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black', linewidth=1.5)
ax.text(0.98, 0.60, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', horizontalalignment='right', bbox=props)

# 旋转x轴标签以提高可读性
plt.xticks(rotation=45, ha='right')

# 调整布局
plt.tight_layout()

# 创建输出目录（如果不存在）
os.makedirs('pdfs', exist_ok=True)

# 保存图形
plt.savefig('pdfs/AGE_USG%_BoxPlot_Regression_MEANS.pdf', 
            dpi=300, bbox_inches='tight')
print("箱线图（基于平均值的回归）已成功保存！")

# 打印详细统计信息
print("\n" + "="*80)
print("各年龄组详细统计")
print("="*80)
print(f"{'年龄':<5} {'数量':<8} {'平均值':<10} {'中位数':<10} {'标准差':<10} {'最小值':<10} {'最大值':<10}")
print("-"*80)

for age in ages:
    age_data = df[df['AGE'] == age]['Performance']
    if len(age_data) > 0:
        print(f"{int(age):<5} {len(age_data):<8} {age_data.mean():<10.2f} {age_data.median():<10.2f} "
              f"{age_data.std():<10.2f} {age_data.min():<10.2f} {age_data.max():<10.2f}")

print("\n" + "="*80)
print("基于平均值的回归分析")
print("="*80)
print(f"二次方程: y = {z[0]:.6f}x² + {z[1]:.6f}x + {z[2]:.6f}")
print(f"R²值: {r_squared:.6f}")
print(f"峰值年龄（来自回归）: {peak_age:.2f} 岁")
print(f"峰值表现（预测值）: {peak_performance:.2f}")

# 根据平均值找出实际表现最佳的年龄
best_age_idx = np.argmax(means)
best_age = age_means[best_age_idx]
best_mean = means[best_age_idx]
print(f"\n实际最佳年龄（按平均值）: {int(best_age)} 岁")
print(f"实际最佳平均表现: {best_mean:.2f}")

# 计算中位数回归以进行对比
medians = [np.median(df[df['AGE'] == age]['Performance'].values) for age in ages]
z_median = np.polyfit(age_means, medians, 2)
p_median = np.poly1d(z_median)
y_pred_median = p_median(age_means)
ss_res_median = np.sum((medians - y_pred_median) ** 2)
ss_tot_median = np.sum((medians - np.mean(medians)) ** 2)
r_squared_median = 1 - (ss_res_median / ss_tot_median)

# 找出中位数回归的峰值年龄
x_median_smooth = np.linspace(min(age_means), max(age_means), 200)
y_median_smooth = p_median(x_median_smooth)
peak_age_median = x_median_smooth[np.argmax(y_median_smooth)]

print("\n" + "="*80)
print("对比：平均值回归 vs 中位数回归")
print("="*80)
print(f"平均值回归 - R²: {r_squared:.6f}, 峰值年龄: {peak_age:.2f}")
print(f"中位数回归 - R²: {r_squared_median:.6f}, 峰值年龄: {peak_age_median:.2f}")
print(f"峰值年龄差异（平均值-中位数）: {peak_age - peak_age_median:.2f} 岁")

plt.close()