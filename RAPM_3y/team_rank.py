import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl

# ===== Global font: Arial =====
mpl.rcParams["font.family"] = "Arial"

# 读取数据
df = pd.read_csv('csv_fold/team_quality.csv')

# 按Normalized_Quality降序排序
df = df.sort_values('Normalized_Quality', ascending=False)

# 创建颜色映射
def get_color(value):
    if value > 0.85:
        return '#1e40af'  # 深蓝色 - 顶级球队
    elif value > 0.7:
        return '#3b82f6'  # 蓝色 - 优秀球队
    elif value > 0.5:
        return '#60a5fa'  # 浅蓝色 - 中上球队
    elif value > 0.3:
        return '#93c5fd'  # 淡蓝色 - 中等球队
    elif value > 0.15:
        return '#dbeafe'  # 很淡的蓝色 - 中下球队
    else:
        return '#f1f5f9'  # 灰蓝色 - 较弱球队

colors = [get_color(val) for val in df['Normalized_Quality']]

# 创建图表
fig, ax = plt.subplots(figsize=(10, 4))

# 绘制柱状图
bars = ax.bar(range(len(df)), df['Normalized_Quality'], color=colors, 
               edgecolor='white', linewidth=0.5)

# 设置x轴
ax.set_xticks(range(len(df)))
ax.set_xticklabels(df['Team'], rotation=45, ha='right', fontsize=11)

# 设置y轴
ax.set_ylabel('Normalized Quality', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.set_yticks(np.arange(0, 1.1, 0.1))

# 添加网格
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)

# 设置标题
#ax.set_title('NBA Team Quality Comparison\nBased on Normalized Quality Metric', fontsize=18, fontweight='bold', pad=20)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1e40af', label='Elite Teams (> 0.85)'),
    Patch(facecolor='#3b82f6', label='Excellent Teams (0.7 - 0.85)'),
    Patch(facecolor='#60a5fa', label='Above Average Teams (0.5 - 0.7)'),
    Patch(facecolor='#93c5fd', label='Average Teams (0.3 - 0.5)'),
    Patch(facecolor='#dbeafe', label='Below Average Teams (0.15 - 0.3)'),
    Patch(facecolor='#f1f5f9', label='Weak Teams (< 0.15)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
          framealpha=0.9, edgecolor='gray')

# 在柱子上方添加数值标签（只显示前5名和后5名）
for i, (idx, row) in enumerate(df.iterrows()):
    if i < 5 or i >= len(df) - 5:
        height = row['Normalized_Quality']
        ax.text(i, height + 0.02, f'{height:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('pdfs/team_quality_chart.pdf', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("Chart saved to team_quality_chart.pdf")

# 显示图表
plt.show()
