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
        win_current: 当前胜场数
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
    # 【关键修正】模型计算的是 ln(Revenue)，需要取指数还原
    log_revenue = (LAMBDA1 * win_current + 
                   LAMBDA2 * np.log(WIN_PREVIOUS) + 
                   LAMBDA3 * T + 
                   LAMBDA4 * MARKET_SIZE + 
                   C)
    revenue = np.exp(log_revenue)  # 转换为实际收入（美元）
    
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
    
    return operating_income  # 移除了+3000的临时调整

def create_3d_visualization(
    win_range: Tuple[float, float] = (0, 82),
    win_steps: int = 50,
    salary_range: Tuple[float, float] = (100e6, 250e6),  # 1亿到2.5亿
    salary_steps: int = 50,
    output_file: str = 'team_financials_3d.png',
    figsize: Tuple[int, int] = (7, 10),
    dpi: int = 150
) -> plt.Figure:
    """
    创建3D财务可视化图表
    
    参数:
        win_range: 胜场数范围 (最小值, 最大值)
        win_steps: 胜场数采样点数
        salary_range: 薪资范围 (最小值, 最大值)
        salary_steps: 薪资采样点数
        output_file: 输出图片文件名
        figsize: 图表尺寸 (宽, 高) 英寸
        dpi: 图片分辨率
    
    返回:
        Matplotlib图表对象
    """
    # 生成网格数据
    win_values = np.linspace(win_range[0], win_range[1], win_steps)
    salary_values = np.linspace(salary_range[0], salary_range[1], salary_steps)
    
    # 创建网格
    Win, Salary = np.meshgrid(win_values, salary_values)
    
    # 计算每个点的营业收入
    Income = np.zeros_like(Win)
    for i in range(salary_steps):
        for j in range(win_steps):
            Income[i, j] = calculate_operating_income(
                win_current=Win[i, j],
                player_expense=Salary[i, j],
            )
    
    # 转换为百万美元以便显示
    Salary_M = Salary / 1e6
    Income_M = Income / 1e6
    
    # 创建图表
    fig = plt.figure(figsize=figsize, facecolor='#0d1117')
    ax = fig.add_subplot(111, projection='3d', facecolor='#0d1117')
    
    # 绘制3D曲面
    surf = ax.plot_surface(
        Win, Salary_M, Income_M,
        cmap='viridis',
        alpha=0.9,
        edgecolor='none',
        linewidth=0,
        antialiased=True,
        rcount=50,
        ccount=50
    )
    
    # 添加等高线投影
    ax.contour(
        Win, Salary_M, Income_M,
        levels=15,
        cmap='viridis',
        alpha=0.4,
        offset=Income_M.min(),
        linewidths=1
    )
    
    # 设置轴标签
    ax.set_xlabel('Current_Win(%)', 
                   fontsize=18, 
                   labelpad=12,
                   color='#e6edf3',
                   weight='bold')
    ax.set_ylabel('Player_Expense(M$)', 
                   fontsize=18, 
                   labelpad=12,
                   color='#e6edf3',
                   weight='bold')
    #ax.set_zlabel('Operating_Income(M$)', fontsize=18, labelpad=12,color='#e6edf3',weight='bold')
    
    # 设置标题
    #title_text = 'NBA team 3d analyze\nwin%-player expense-operating income'
    #ax.set_title(title_text, fontsize=20, pad=20,color='#58a6ff',weight='bold')
    
    # 设置轴刻度颜色
    ax.tick_params(axis='x', colors='#8b949e', labelsize=10)
    ax.tick_params(axis='y', colors='#8b949e', labelsize=10)
    ax.tick_params(axis='z', colors='#8b949e', labelsize=10)
    
    # 设置网格
    ax.grid(True, alpha=0.3, color='#30363d', linestyle='-', linewidth=0.5)
    
    # 设置背景颜色
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#30363d')
    ax.yaxis.pane.set_edgecolor('#30363d')
    ax.zaxis.pane.set_edgecolor('#30363d')
    
    # 设置视角
    ax.view_init(elev=25, azim=45)
    
    # 添加颜色条
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('operating income (M$)', 
                   rotation=270, 
                   labelpad=20,
                   color='#e6edf3',
                   fontsize=16,
                   #fontfamily='monospace',
                   weight='bold')
    cbar.ax.tick_params(labelsize=10, colors='#8b949e')
    cbar.outline.set_edgecolor('#30363d')
    cbar.ax.set_facecolor('#161b22')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(
        output_file,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='#0d1117',
        edgecolor='none'
    )
    
    print(f"✓ 3D可视化已生成: {output_file}")
    print(f"  - 图片尺寸: {figsize[0]}×{figsize[1]} 英寸")
    print(f"  - 分辨率: {dpi} DPI")
    print(f"  - 胜场数范围: {win_range[0]}-{win_range[1]} ({win_steps}个采样点)")
    print(f"  - 薪资范围: ${salary_range[0]/1e6:.1f}M-${salary_range[1]/1e6:.1f}M ({salary_steps}个采样点)")
    
    return fig

def create_multiple_views(
    win_range: Tuple[float, float] = (0, 82),
    salary_range: Tuple[float, float] = (100e6, 250e6),
    output_file: str = 'team_financials_multiple_views.png'
):
    """
    创建多视角的3D可视化图表
    """
    # 生成数据
    win_steps, salary_steps = 50, 50
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
    
    # 创建2x2子图
    fig = plt.figure(figsize=(20, 16), facecolor='#0d1117')
    
    # 定义四个视角
    views = [
        (25, 45, 'Standard View'),
        (10, 135, 'Side View'),
        (60, 45, 'Top View'),
        (5, 225, 'Back View')
    ]
    
    for idx, (elev, azim, view_name) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d', facecolor='#0d1117')
        
        # 绘制曲面
        surf = ax.plot_surface(
            Win, Salary_M, Income_M,
            cmap='viridis',
            alpha=0.9,
            edgecolor='none',
            linewidth=0,
            antialiased=True
        )
        
        # 添加等高线
        ax.contour(
            Win, Salary_M, Income_M,
            levels=10,
            cmap='viridis',
            alpha=0.3,
            offset=Income_M.min(),
            linewidths=1
        )
        
        # 设置标签
        ax.set_xlabel('Win(%)', fontsize=11, color='#e6edf3', labelpad=8)
        ax.set_ylabel('Salary(M$)', fontsize=11, color='#e6edf3', labelpad=8)
        ax.set_zlabel('Income(M$)', fontsize=11, color='#e6edf3', labelpad=8)
        ax.set_title(view_name, fontsize=14, color='#58a6ff', pad=15, weight='bold')
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 设置样式
        ax.tick_params(axis='x', colors='#8b949e', labelsize=9)
        ax.tick_params(axis='y', colors='#8b949e', labelsize=9)
        ax.tick_params(axis='z', colors='#8b949e', labelsize=9)
        ax.grid(True, alpha=0.2, color='#30363d')
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#30363d')
        ax.yaxis.pane.set_edgecolor('#30363d')
        ax.zaxis.pane.set_edgecolor('#30363d')
    
    # 总标题
    fig.suptitle(
        'NBA Team Financial Analysis - Multiple Views',
        fontsize=22,
        color='#58a6ff',
        weight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    
    print(f"✓ 多视角图表已生成: {output_file}")
    
    return fig

if __name__ == "__main__":
    print("=" * 70)
    print("NBA球队财务3D可视化工具（修正版）")
    print("=" * 70)
    
    # 示例1: 标准视角
    print("\n[1/3] 生成标准3D视图...")
    fig1 = create_3d_visualization(
        output_file='team_financials_3d_standard.pdf'
    )
    plt.close(fig1)
    
    # 示例2: 自定义参数
    print("\n[2/3] 生成自定义参数视图...")
    fig2 = create_3d_visualization(
        win_range=(20, 70),                 # 只看20-70胜
        salary_range=(120e6, 200e6),        # 薪资1.2亿-2亿
        output_file='team_financials_3d_custom.pdf'
    )
    plt.close(fig2)
    
    # 示例3: 多视角
    print("\n[3/3] 生成多视角视图...")
    fig3 = create_multiple_views(
        output_file='team_financials_multiple_views.pdf'
    )
    plt.close(fig3)
    
    print("\n" + "=" * 70)
    print("✓ 所有图表生成完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  1. team_financials_3d_standard.pdf - 标准参数视图")
    print("  2. team_financials_3d_custom.pdf - 自定义参数视图")
    print("  3. team_financials_multiple_views.pdf - 多视角展示")
    print("\n主要修正:")
    print("  ✓ 添加 np.exp() 将 ln(Revenue) 转换为 Revenue")
    print("  ✓ 移除了 +3000 的临时调整")
    print("  ✓ 统一了单位（美元）")

