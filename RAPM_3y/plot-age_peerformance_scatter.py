import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams["font.family"] = "Arial"
# 读取数据
input_file = 'csv_fold/AGE_USG%.csv'
df = pd.read_csv(input_file)

# 提取年龄和表现数据
age = df['AGE'].values
performance = df['Performance'].values

# 创建二次回归模型
X = age.reshape(-1, 1)
y = performance

# 使用numpy拟合二次多项式
coefficients = np.polyfit(age, performance, 2)
poly_function = np.poly1d(coefficients)

# 生成用于绘制曲线的平滑数据点
age_smooth = np.linspace(age.min(), age.max(), 300)
performance_smooth = poly_function(age_smooth)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制散点图
plt.scatter(age, performance, alpha=0.5, s=30, c='steelblue', edgecolors='none', label='point')

# 绘制二次回归曲线
plt.plot(age_smooth, performance_smooth, 'r-', linewidth=2.5, label='quadratic regression curve')

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 添加标签和标题
plt.xlabel('AGE', fontsize=14, fontweight='bold')
plt.ylabel('USG%', fontsize=14, fontweight='bold')

# 添加图例
plt.legend(fontsize=12, loc='upper right')

# 显示回归方程
equation_text = f'function: y = {coefficients[0]:.4f}x^2 + {coefficients[1]:.4f}x + {coefficients[2]:.4f}'
plt.text(0.02, 0.98, equation_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 计算R²值
y_pred = poly_function(age)
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - np.mean(y)) ** 2)
r_squared = 1 - (ss_res / ss_tot)


# 显示统计信息
print(f"\n二次回归分析结果:")
print(f"回归方程: y = {coefficients[0]:.4f}x² + {coefficients[1]:.4f}x + {coefficients[2]:.4f}")
print(f"R² 值: {r_squared:.4f}")
print(f"\n数据集统计:")
print(f"总数据点数: {len(age)}")
print(f"年龄范围: {age.min():.0f} - {age.max():.0f}")
print(f"使用率范围: {performance.min():.1f}% - {performance.max():.1f}%")
print(f"平均年龄: {age.mean():.2f}")
print(f"平均使用率: {performance.mean():.2f}%")

# 找出峰值年龄（二次函数的顶点）
peak_age = -coefficients[1] / (2 * coefficients[0])
peak_performance = poly_function(peak_age)
print(f"\n根据回归曲线:")
print(f"使用率峰值年龄: {peak_age:.2f} 岁")
print(f"预测峰值使用率: {peak_performance:.2f}%")

# 显示R²值
r2_text = f'R^2 = {r_squared:.4f}; Maxium: age = {peak_age:.4f},usg% = {peak_performance:.4f}'
plt.text(0.02, 0.90, r2_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 调整布局
plt.tight_layout()

# 保存图形
output_file = 'pdfs/age_performance_scatter_plot.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"图形已保存至: {output_file}")

plt.show()
