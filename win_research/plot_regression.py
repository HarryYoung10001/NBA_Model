import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据
df = pd.read_csv('regression.csv')
X = df[['RAPM']].values
y = df['win_pct'].values

# 线性回归
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)
r2 = r2_score(y, y_pred)

# 创建图形，尺寸3.4*2.4英寸
fig, ax = plt.subplots(figsize=(3.4, 2.4))

# 散点图
ax.scatter(X, y, alpha=0.6, s=20, edgecolors='black', linewidth=0.3, color='#3498db')

# 回归线
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = lr.predict(X_line)
ax.plot(X_line, y_line, 'r-', linewidth=1.5, label=f'$R^2$ = {r2:.4f}')

# 设置标签和标题
ax.set_xlabel('RAPM-TEAM', fontsize=9, fontweight='bold')
ax.set_ylabel('Win Percentage', fontsize=9, fontweight='bold')

# 添加图例
ax.legend(fontsize=8, loc='lower right')

# 网格
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('regression_plot.pdf', dpi=300, bbox_inches='tight')
print(f"散点图已保存！")
print(f"线性回归方程: win_pct = {lr.intercept_:.4f} + {lr.coef_[0]:.4f} × RAPM")
print(f"R² = {r2:.4f}")

