import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv('TQ_stat/TQ_stat_with_zscore.csv')

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 定义颜色方案
years = sorted(df['year'].unique())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5种不同颜色

# 为每个年份绘制散点
for i, year in enumerate(years):
    year_data = df[df['year'] == year]
    ax.scatter(year_data['Normalized_TQ_Z_score'], year_data['win_pct'], 
               c=colors[i], label=str(year), alpha=0.7, s=100, edgecolors='black', linewidth=0.5)

# 添加趋势线（整体数据）
z = np.polyfit(df['Normalized_TQ_Z_score'], df['win_pct'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df['Normalized_TQ_Z_score'].min(), df['Normalized_TQ_Z_score'].max(), 100)
ax.plot(x_trend, p(x_trend), 'k--', alpha=0.5, linewidth=2, label='Trend line')

# 设置标签和标题
ax.set_xlabel('TQ Z-score', fontsize=14, fontweight='bold')
ax.set_ylabel('Win Percentage', fontsize=14, fontweight='bold')
ax.set_title('TQ Z-score vs Win Percentage by Year', fontsize=16, fontweight='bold')

# 添加网格
ax.grid(True, alpha=0.3, linestyle='--')

# 添加图例
ax.legend(loc='upper left', fontsize=11, framealpha=0.9)

# 调整布局
plt.tight_layout()

# 保存图形
output_path = 'pdfs/TQ_zscore_vs_winpct.pdf'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图形已保存到: {output_path}")

# 计算相关系数
correlation = df['Normalized_TQ_Z_score'].corr(df['win_pct'])
print(f"\nTQ Z-score 和 Win Percentage 的相关系数: {correlation:.4f}")

# 显示每年的相关系数
print("\n各年份的相关系数:")
for year in years:
    year_data = df[df['year'] == year]
    year_corr = year_data['Normalized_TQ_Z_score'].corr(year_data['win_pct'])
    print(f"  {year}: {year_corr:.4f}")

plt.show()

