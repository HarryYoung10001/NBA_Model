import numpy as np
from typing import List, Dict

def calculate_team_financials_simplified(
    win_current: float,                    # 当前胜场数
    player_expense: float,                 # 薪资总支出 ∑Sal_{p,i}
    total_league_revenue: float,           # 联盟总收入
    salary_cap_2023_24: float,            # 2023-24赛季基准薪资帽
    penalty_constant: float = 0.0         # Apron惩罚常数（可选）
) -> Dict:
    """
    简化版球队财务计算
    
    参数:
        win_current: 当前胜场数 Win_i(t)
        player_expense: 薪资总支出 ∑Sal_{p,i}
        total_league_revenue: 联盟总收入
        salary_cap_2023_24: 2023-24赛季基准薪资帽
        penalty_constant: Apron惩罚常数（默认0）
    
    返回:
        包含所有财务指标的字典
    """
    # ========== 常数定义 ==========
    WIN_PREVIOUS = 0.62195122
    LAMBDA1 = 0.23237020317768445
    LAMBDA2 = 0.04964080705789893
    LAMBDA3 = 0.06982744476580202
    LAMBDA4 = 0.9573219880279853
    C = -135.79433806768216
    T = 2026
    MARKET_SIZE = 0.660672715
    
    # ========== 1. 计算收入 ==========
    revenue = (LAMBDA1 * win_current + 
               LAMBDA2 * np.log(WIN_PREVIOUS) + 
               LAMBDA3 * T + 
               LAMBDA4 * MARKET_SIZE + 
               C)
    
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
        
        # 超过4W的部分
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
    
    # ========== 8. 检查薪资下限 ==========
    meets_floor = player_expense >= 0.9 * salary_cap
    
    return operating_income


# ========== 使用示例 ==========
if __name__ == "__main__":
    result = calculate_team_financials_simplified(
        win_current=50,                    # 假设赢了50场
        player_expense=150e6,              # 薪资总支出1.5亿
        total_league_revenue=10e9,         # 联盟总收入100亿
        salary_cap_2023_24=136e6,         # 2023-24薪资帽1.36亿
        penalty_constant=5e6               # Apron惩罚500万
    )
    
    print("财务计算结果：")
    for key, value in result.items():
        if isinstance(value, bool):
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:,.2f}")
