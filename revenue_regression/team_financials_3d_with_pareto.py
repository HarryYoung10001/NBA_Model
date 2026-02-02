import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple
import matplotlib as mpl
from pathlib import Path

# ===== Global font: Arial =====
mpl.rcParams["font.family"] = "Arial"

def calculate_operating_income(
    win_current: float,
    player_expense: float,
) -> float:
    """
    计算球队营业收入（修正版）
    
    参数:
        win_current: 当前胜率 (0-1)
        player_expense: 薪资总支出（单位：美元）
    
    返回:
        营业收入 (美元)
    """
    # ========== 常数定义 ==========
    total_league_revenue = 10200e6      # 102亿美元
    salary_cap_2023_24 = 136.021e6      # 1.36021亿美元
    penalty_constant = 200
    
    # 回归模型参数
    WIN_PREVIOUS = 0.62195122
    LAMBDA1 = 0.23237020317768445
    LAMBDA2 = 0.04964080705789893
    LAMBDA3 = 0.06982744476580202
    LAMBDA4 = 0.9573219880279853
    C = -135.79433806768216
    T = 2026
    MARKET_SIZE = 0.660672715
    
    # ========== 1. 计算收入 ==========
    # 【关键修正】模型计算的是 ln(Revenue_in_M$)，需要取指数还原
    log_revenue = (LAMBDA1 * win_current + 
                   LAMBDA2 * np.log(WIN_PREVIOUS) + 
                   LAMBDA3 * T + 
                   LAMBDA4 * MARKET_SIZE + 
                   C)
    revenue_M = np.exp(log_revenue)  # 收入（单位：百万美元）
    revenue = revenue_M * 1e6  # 转换为美元
    
    # ========== 2. 计算薪资帽 ==========
    salary_cap = (total_league_revenue * 0.51) / 30
    
    # ========== 3. 计算奢侈税线 ==========
    luxury_tax_line = 1.2 * salary_cap
    
    # ========== 4. 计算税级宽度 ==========
    bracket_width = 5e6 * (salary_cap / salary_cap_2023_24)
    
    # ========== 5. 计算奢侈税 ==========
    X = player_expense - luxury_tax_line  # 应税金额
    
    luxury_tax = 0.0
    if X > 0:
        tax_rates = [1.50, 1.75, 2.50, 3.25]
        
        # 前4个税级
        for j in range(4):
            if X > j * bracket_width:
                taxable = min(X - j * bracket_width, bracket_width)
                luxury_tax += taxable * tax_rates[j]
        
        # 超过4个税级的部分
        j = 4
        while X > j * bracket_width:
            taxable = min(X - j * bracket_width, bracket_width)
            tax_rate = 3.25 + 0.5 * (j - 3)
            luxury_tax += taxable * tax_rate
            j += 1
    
    # ========== 6. 计算Apron阈值和惩罚 ==========
    apron1 = 1.3 * salary_cap
    apron2 = 1.5 * salary_cap
    
    apron_penalty = 0.0
    adjusted_revenue = revenue
    
    if revenue > apron1:
        apron_penalty = penalty_constant
    if revenue > apron2:
        adjusted_revenue = apron2
    
    # ========== 7. 计算总成本和营业收入 ==========
    total_cost = player_expense + luxury_tax + apron_penalty
    operating_income = adjusted_revenue - total_cost
    
    return operating_income

def create_dual_view_with_pareto(
    win_range: Tuple[float, float] = (0.2, 0.85),  # 胜率范围
    salary_range: Tuple[float, float] = (70e6, 280e6),
    output_file: str = 'team_financials_dual_view.pdf'
):
    """
    创建双视角的3D可视化图表，标注Pareto最优点
    """
    # Pareto最优解（从之前的分析结果）
    pareto_points = {
        'A_Win': {'win_pct': 0.744, 'expense_M': 193.7, 'income_M': 66.4},
        'B_Income': {'win_pct': 0.510, 'expense_M': 72.0, 'income_M': 188.1},
        'C_Balance': {'win_pct': 0.650, 'expense_M': 107.8, 'income_M': 152.3}
    }
    
    # 生成数据
    win_steps, salary_steps = 60, 60
    win_values = np.linspace(win_range[0], win_range[1], win_steps)
    salary_values = np.linspace(salary_range[0], salary_range[1], salary_steps)
    Win, Salary = np.meshgrid(win_values, salary_values)
    
    Income = np.zeros_like(Win)
    for i in range(salary_steps):
        for j in range(win_steps):
            Income[i, j] = calculate_operating_income(
                win_current=Win[i, j],
                player_expense=Salary[i, j],
            )
    
    Salary_M = Salary / 1e6
    Income_M = Income / 1e6
    
    # 创建1x2子图，紧凑布局
    fig = plt.figure(figsize=(16, 7), facecolor='#0d1117')
    
    # 定义两个视角
    views = [
        (10, 135, 'Side View'),
        (5, 225, 'Back View')
    ]
    
    for idx, (elev, azim, view_name) in enumerate(views, 1):
        ax = fig.add_subplot(1, 2, idx, projection='3d', facecolor='#0d1117')
        
        # 绘制曲面
        surf = ax.plot_surface(
            Win, Salary_M, Income_M,
            cmap='viridis',
            alpha=0.85,
            edgecolor='none',
            linewidth=0,
            antialiased=True,
            rcount=60,
            ccount=60
        )
        
        # 添加等高线
        ax.contour(
            Win, Salary_M, Income_M,
            levels=12,
            cmap='viridis',
            alpha=0.35,
            offset=Income_M.min(),
            linewidths=1
        )
        
        # ===== 标注Pareto最优点 =====
        point_colors = {'A_Win': 'red', 'B_Income': 'gold', 'C_Balance': 'lime'}
        point_labels = {'A_Win': 'Win-Max', 'B_Income': 'Income-Max', 'C_Balance': 'Balanced'}
        point_markers = {'A_Win': '*', 'B_Income': 'D', 'C_Balance': 'o'}
        
        for key, point in pareto_points.items():
            ax.scatter(
                point['win_pct'],
                point['expense_M'],
                point['income_M'],
                color=point_colors[key],
                s=250,
                marker=point_markers[key],
                edgecolors='white',
                linewidths=2,
                zorder=10,
                label=point_labels[key]
            )
        
        # 设置轴标签
        ax.set_xlabel('Win%', fontsize=12, color='#e6edf3', labelpad=8, weight='bold')
        ax.set_ylabel('Expense (M$)', fontsize=12, color='#e6edf3', labelpad=8, weight='bold')
        ax.set_zlabel('Income (M$)', fontsize=12, color='#e6edf3', labelpad=8, weight='bold')
        
        # 子图标题
        ax.text2D(0.5, 0.95, view_name, 
                  transform=ax.transAxes,
                  fontsize=13, 
                  color='#58a6ff', 
                  ha='center',
                  weight='bold')
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 设置样式
        ax.tick_params(axis='x', colors='#8b949e', labelsize=9)
        ax.tick_params(axis='y', colors='#8b949e', labelsize=9)
        ax.tick_params(axis='z', colors='#8b949e', labelsize=9)
        ax.grid(True, alpha=0.25, color='#30363d', linestyle='-', linewidth=0.5)
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#30363d')
        ax.yaxis.pane.set_edgecolor('#30363d')
        ax.zaxis.pane.set_edgecolor('#30363d')
        
        # 添加图例
        legend = ax.legend(loc='upper left', fontsize=9, framealpha=0.7, facecolor='#161b22', edgecolor='#30363d')
        for text in legend.get_texts():
            text.set_color('#e6edf3')
    
    # 紧凑布局，不要总标题
    plt.tight_layout(pad=1.5)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    
    print(f"✓ 双视角图表已生成: {output_file}")
    print(f"\n标注的Pareto最优点:")
    for key, point in pareto_points.items():
        print(f"  {point_labels[key]:12s}: Win%={point['win_pct']:.3f}, "
              f"Expense=${point['expense_M']:.1f}M, Income=${point['income_M']:.1f}M")
    
    return fig

if __name__ == "__main__":
    print("=" * 70)
    print("NBA Team Financial 3D Visualization with Pareto Optimal Points")
    print("=" * 70)
    
    # 生成双视角图
    print("\nGenerating dual-view visualization...")
    fig = create_dual_view_with_pareto(
        output_file='team_financials_dual_view.pdf'
    )
    plt.close(fig)
    
    print("\n" + "=" * 70)
    print("✓ Visualization complete!")
    print("=" * 70)
    print("\nGenerated file:")
    print("  • team_financials_dual_view.pdf - Side & Back views with Pareto points")
    print("\nKey features:")
    print("  ✓ Compact dual-view layout (no main title)")
    print("  ✓ Pareto optimal points marked with distinct colors/shapes")
    print("  ✓ Revenue calculation corrected (exp transformation)")
