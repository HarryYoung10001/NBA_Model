import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d
from sklearn.isotonic import IsotonicRegression
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings('ignore')

# ===== Global font: Arial =====
mpl.rcParams["font.family"] = "Arial"

# ============================================================================
# 用户提供的营业收入计算函数
# ============================================================================
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


# ============================================================================
# 单调分位数包线构造函数
# ============================================================================
def build_isotonic_quantile_band(df, q_lower=0.10, q_upper=0.90, n_bins=50):
    """
    构造单调分位数包线
    
    参数:
        df: DataFrame，需包含 'Player Expenses(M$)' 和 'win_pct' 列
        q_lower: 下分位数（例如 0.10 表示排除最低10%）
        q_upper: 上分位数（例如 0.90 表示排除最高10%）
        n_bins: 分箱数量
    
    返回:
        f_lower: 下包线函数
        f_upper: 上包线函数
        x_min: expense最小值
        x_max: expense最大值
    """
    X = df['Player Expenses(M$)'].values
    Y = df['win_pct'].values
    
    # 确定expense范围
    x_min, x_max = X.min(), X.max()
    
    # 分箱
    bins = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 计算每个箱的分位数
    quantiles_lower = []
    quantiles_upper = []
    valid_centers = []
    
    for i in range(n_bins):
        mask = (X >= bins[i]) & (X < bins[i+1])
        if mask.sum() >= 3:  # 至少3个点才计算分位数
            y_bin = Y[mask]
            quantiles_lower.append(np.quantile(y_bin, q_lower))
            quantiles_upper.append(np.quantile(y_bin, q_upper))
            valid_centers.append(bin_centers[i])
    
    quantiles_lower = np.array(quantiles_lower)
    quantiles_upper = np.array(quantiles_upper)
    valid_centers = np.array(valid_centers)
    
    # 单调回归
    iso_lower = IsotonicRegression(increasing=True)
    iso_upper = IsotonicRegression(increasing=True)
    
    y_lower_mono = iso_lower.fit_transform(valid_centers, quantiles_lower)
    y_upper_mono = iso_upper.fit_transform(valid_centers, quantiles_upper)
    
    # 插值成函数
    f_lower = interp1d(valid_centers, y_lower_mono, 
                       kind='linear', bounds_error=False, 
                       fill_value=(y_lower_mono[0], y_lower_mono[-1]))
    f_upper = interp1d(valid_centers, y_upper_mono, 
                       kind='linear', bounds_error=False, 
                       fill_value=(y_upper_mono[0], y_upper_mono[-1]))
    
    return f_lower, f_upper, x_min, x_max, valid_centers, y_lower_mono, y_upper_mono


# ============================================================================
# Pareto最优筛选
# ============================================================================
def get_pareto_optimal(candidates, obj_keys=['win_pct', 'income']):
    """
    筛选Pareto最优解（两个目标都要maximize）
    
    参数:
        candidates: list of dict，每个dict包含目标值
        obj_keys: 要优化的目标key列表
    
    返回:
        pareto_front: Pareto最优解列表
    """
    n = len(candidates)
    is_pareto = np.ones(n, dtype=bool)
    
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            # 如果j在所有目标上都不弱于i，且至少一个严格更好
            dominated = True
            at_least_one_better = False
            for key in obj_keys:
                if candidates[j][key] < candidates[i][key]:
                    dominated = False
                    break
                if candidates[j][key] > candidates[i][key]:
                    at_least_one_better = True
            
            if dominated and at_least_one_better:
                is_pareto[i] = False
                break
    
    return [candidates[i] for i in range(n) if is_pareto[i]]


# ============================================================================
# 找Pareto前沿的膝点（knee point）
# ============================================================================
def find_knee_point(pareto_front):
    """
    找Pareto前沿的膝点（最大曲率点）
    
    使用归一化欧氏距离到理想点
    """
    if len(pareto_front) <= 2:
        return pareto_front[0]
    
    # 提取目标值
    wins = np.array([p['win_pct'] for p in pareto_front])
    incomes = np.array([p['income'] for p in pareto_front])
    
    # 归一化到[0,1]
    win_norm = (wins - wins.min()) / (wins.max() - wins.min() + 1e-10)
    inc_norm = (incomes - incomes.min()) / (incomes.max() - incomes.min() + 1e-10)
    
    # 理想点是(1, 1)，计算距离
    distances = np.sqrt((1 - win_norm)**2 + (1 - inc_norm)**2)
    
    # 返回距离最小的点
    knee_idx = np.argmin(distances)
    return pareto_front[knee_idx]


# ============================================================================
# 主分析流程
# ============================================================================
def main():
    # 1. 读取数据
    print("=" * 80)
    print("NBA球队优化分析：基于单调分位数包线的双目标优化")
    print("=" * 80)
    print()
    
    df = pd.read_csv('revenue_regression/merged_team_year_stats_2016_2024.csv')
    print(f"数据概况：{len(df)} 个赛季样本（{df['year'].min()}-{df['year'].max()}）")
    print(f"Expense范围：${df['Player Expenses(M$)'].min():.1f}M - ${df['Player Expenses(M$)'].max():.1f}M")
    print(f"Win%范围：{df['win_pct'].min():.3f} - {df['win_pct'].max():.3f}")
    print()
    
    # 2. 构造80%单调分位数包线
    print("构造可行域：80%单调分位数包线（排除极端10% + 10%）")
    f_lower, f_upper, x_min, x_max, centers, y_lower, y_upper = \
        build_isotonic_quantile_band(df, q_lower=0.10, q_upper=0.90, n_bins=30)
    
    coverage = ((df['win_pct'] >= f_lower(df['Player Expenses(M$)'])) & 
                (df['win_pct'] <= f_upper(df['Player Expenses(M$)']))).mean()
    print(f"包线覆盖率：{coverage*100:.1f}% 的历史数据点")
    print()
    
    # 3. 生成边界候选点
    print("生成边界候选点...")
    x_grid = np.linspace(x_min, x_max, 800)
    candidates = []
    
    for x in x_grid:
        # 上包线：追求高胜率
        candidates.append({
            'expense_M': x,
            'win_pct': float(f_upper(x)),
            'source': 'upper_bound'
        })
        # 下包线：低配运营
        candidates.append({
            'expense_M': x,
            'win_pct': float(f_lower(x)),
            'source': 'lower_bound'
        })
    
    print(f"候选点总数：{len(candidates)} 个")
    print()
    
    # 4. 计算每个候选点的营业收入
    print("计算营业收入（调用用户函数）...")
    for c in candidates:
        expense_usd = c['expense_M'] * 1e6  # M$ -> $
        c['income'] = calculate_operating_income(c['win_pct'], expense_usd)
        c['income_M'] = c['income'] / 1e6  # 转为百万美元便于展示
    
    # 检查收入范围
    incomes = [c['income_M'] for c in candidates]
    print(f"营业收入范围：${min(incomes):.1f}M - ${max(incomes):.1f}M")
    print()
    
    # 5. Pareto筛选
    print("Pareto最优筛选...")
    pareto_front = get_pareto_optimal(candidates, obj_keys=['win_pct', 'income'])
    print(f"Pareto前沿点数：{len(pareto_front)} 个（从{len(candidates)}个候选中筛选）")
    print()
    
    # 6. 找关键解
    print("=" * 80)
    print("关键最优解")
    print("=" * 80)
    print()
    
    # 6.1 赢球优先
    best_win = max(pareto_front, key=lambda p: p['win_pct'])
    print("【方案A：赢球优先】")
    print(f"  薪资支出：${best_win['expense_M']:.1f}M")
    print(f"  预期胜率：{best_win['win_pct']:.3f} ({best_win['win_pct']*82:.1f}胜)")
    print(f"  营业收入：${best_win['income_M']:.1f}M")
    print(f"  策略定位：{best_win['source']}")
    print()
    
    # 6.2 挣钱优先
    best_income = max(pareto_front, key=lambda p: p['income'])
    print("【方案B：收入优先】")
    print(f"  薪资支出：${best_income['expense_M']:.1f}M")
    print(f"  预期胜率：{best_income['win_pct']:.3f} ({best_income['win_pct']*82:.1f}胜)")
    print(f"  营业收入：${best_income['income_M']:.1f}M")
    print(f"  策略定位：{best_income['source']}")
    print()
    
    # 6.3 膝点（平衡解）
    knee = find_knee_point(pareto_front)
    print("【方案C：平衡解（膝点）】")
    print(f"  薪资支出：${knee['expense_M']:.1f}M")
    print(f"  预期胜率：{knee['win_pct']:.3f} ({knee['win_pct']*82:.1f}胜)")
    print(f"  营业收入：${knee['income_M']:.1f}M")
    print(f"  策略定位：{knee['source']}")
    print()
    
    # 7. 可视化
    print("生成可视化...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：可行域 + 历史数据
    ax1 = axes[0]
    ax1.scatter(df['Player Expenses(M$)'], df['win_pct'], 
                alpha=0.3, s=20, c='gray', label='Historical Data (2016-2024)')
    
    # 绘制包线
    x_plot = np.linspace(x_min, x_max, 200)
    ax1.plot(x_plot, f_lower(x_plot), 'b--', linewidth=2, label='Lower Bound (10% quantile)')
    ax1.plot(x_plot, f_upper(x_plot), 'r--', linewidth=2, label='Upper Bound (90% quantile)')
    
    # 填充可行域
    ax1.fill_between(x_plot, f_lower(x_plot), f_upper(x_plot), 
                     alpha=0.15, color='green', label='Feasible Region (80% coverage)')
    
    # 标注关键解
    ax1.scatter(best_win['expense_M'], best_win['win_pct'], 
               s=200, c='red', marker='*', edgecolors='black', linewidths=1.5,
               label='Solution A: Win-Maximizing', zorder=5)
    ax1.scatter(best_income['expense_M'], best_income['win_pct'], 
               s=200, c='gold', marker='*', edgecolors='black', linewidths=1.5,
               label='Solution B: Income-Maximizing', zorder=5)
    ax1.scatter(knee['expense_M'], knee['win_pct'], 
               s=200, c='lime', marker='*', edgecolors='black', linewidths=1.5,
               label='Solution C: Balanced (Knee)', zorder=5)
    
    ax1.set_xlabel('Player Expenses (M$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Win%', fontsize=12, fontweight='bold')
    ax1.set_title('Feasible Region: 80% Monotonic Quantile Band', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 右图：Pareto前沿
    ax2 = axes[1]
    
    # 所有候选点
    ax2.scatter([c['win_pct'] for c in candidates], 
                [c['income_M'] for c in candidates],
                alpha=0.1, s=10, c='gray', label='Boundary Candidates')
    
    # Pareto前沿
    pareto_wins = [p['win_pct'] for p in pareto_front]
    pareto_incomes = [p['income_M'] for p in pareto_front]
    ax2.scatter(pareto_wins, pareto_incomes, 
                s=30, c='blue', alpha=0.6, label='Pareto Front')
    
    # 排序后连线
    pareto_sorted = sorted(pareto_front, key=lambda p: p['win_pct'])
    ax2.plot([p['win_pct'] for p in pareto_sorted], 
             [p['income_M'] for p in pareto_sorted],
             'b-', linewidth=1.5, alpha=0.5)
    
    # 标注关键解
    ax2.scatter(best_win['win_pct'], best_win['income_M'], 
               s=200, c='red', marker='*', edgecolors='black', linewidths=1.5,
               label='Solution A: Win-Maximizing', zorder=5)
    ax2.scatter(best_income['win_pct'], best_income['income_M'], 
               s=200, c='gold', marker='*', edgecolors='black', linewidths=1.5,
               label='Solution B: Income-Maximizing', zorder=5)
    ax2.scatter(knee['win_pct'], knee['income_M'], 
               s=200, c='lime', marker='*', edgecolors='black', linewidths=1.5,
               label='Solution C: Balanced (Knee)', zorder=5)
    
    ax2.set_xlabel('Win%', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Operating Income (M$)', fontsize=12, fontweight='bold')
    ax2.set_title('Pareto Front: Bi-Objective Trade-off', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pdfs/nba_optimization_result.pdf', dpi=150, bbox_inches='tight')
    print("可视化已保存！")
    print()
    
    # 8. 导出详细结果
    pareto_df = pd.DataFrame(pareto_front)
    pareto_df = pareto_df.sort_values('win_pct', ascending=False)
    pareto_df.to_csv('csv_fold/pareto_solutions.csv', index=False)
    print(f"Pareto前沿详细结果已导出（{len(pareto_df)}个最优解）")
    
    # 9. 生成决策报告
    report = []
    report.append("=" * 80)
    report.append("NBA球队优化决策报告")
    report.append("=" * 80)
    report.append("")
    report.append("【分析方法】")
    report.append("  可行域：80%单调分位数包线（基于2016-2024历史数据）")
    report.append("  优化目标：maximize Win% + maximize Operating Income")
    report.append(f"  候选点数：{len(candidates)} 个边界点")
    report.append(f"  Pareto解：{len(pareto_front)} 个非支配解")
    report.append("")
    report.append("【最优方案对比】")
    report.append("")
    report.append("方案A（赢球优先）：")
    report.append(f"  ├─ 薪资支出：${best_win['expense_M']:.1f}M")
    report.append(f"  ├─ 预期胜率：{best_win['win_pct']:.3f} ≈ {best_win['win_pct']*82:.0f}胜")
    report.append(f"  ├─ 营业收入：${best_win['income_M']:.1f}M")
    report.append(f"  └─ 适用场景：追求冠军，不计成本")
    report.append("")
    report.append("方案B（收入优先）：")
    report.append(f"  ├─ 薪资支出：${best_income['expense_M']:.1f}M")
    report.append(f"  ├─ 预期胜率：{best_income['win_pct']:.3f} ≈ {best_income['win_pct']*82:.0f}胜")
    report.append(f"  ├─ 营业收入：${best_income['income_M']:.1f}M")
    report.append(f"  └─ 适用场景：重建期，优先盈利")
    report.append("")
    report.append("方案C（平衡解/膝点）：")
    report.append(f"  ├─ 薪资支出：${knee['expense_M']:.1f}M")
    report.append(f"  ├─ 预期胜率：{knee['win_pct']:.3f} ≈ {knee['win_pct']*82:.0f}胜")
    report.append(f"  ├─ 营业收入：${knee['income_M']:.1f}M")
    report.append(f"  └─ 适用场景：竞争性运营，双目标兼顾")
    report.append("")
    report.append("【决策建议】")
    report.append("  1. 若球队处于夺冠窗口期，选择方案A")
    report.append("  2. 若球队处于重建期或财务压力大，选择方案B")
    report.append("  3. 若希望长期可持续发展，选择方案C（推荐）")
    report.append("")
    report.append("【技术说明】")
    report.append("  - 所有方案均在历史可行域内（80%覆盖率）")
    report.append("  - 所有方案均为Pareto最优（无法同时改进两个目标）")
    report.append("  - 膝点通过归一化欧氏距离到理想点(1,1)计算")
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    with open('decision_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    
    return {
        'pareto_front': pareto_front,
        'best_win': best_win,
        'best_income': best_income,
        'knee': knee,
        'f_lower': f_lower,
        'f_upper': f_upper,
        'x_range': (x_min, x_max)
    }


if __name__ == '__main__':
    results = main()
