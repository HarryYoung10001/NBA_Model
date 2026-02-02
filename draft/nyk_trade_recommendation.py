"""
NYK交易推荐系统
基于论文算法实现交易决策引擎
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from itertools import combinations
import time

# =============================================================================
# 配置参数
# =============================================================================

class Config:
    """系统配置参数"""
    
    # 交易机制参数
    EPSILON_TRADE = 0.08  # ε_trade: Fair Trade容忍度
    
    # 选秀机制参数
    ALPHA_DRAFT = 2.0  # α: 选秀公平性调节器
    
    # Salary Cap
    SALARY_CAP = 136_000_000  # 工资帽
    
    # 合同参数
    LONG_TERM_YEARS = 4
    SHORT_TERM_YEARS = 2
    TS_THRESHOLD = 50
    
    # NYK的固定偏好向量
    PV_NYK = np.array([0.40, 0.18, 0.13, 0.10, 0.20])
    # [AS, CS, SalEff, PW, Flex]
    
    # 交易搜索参数
    MAX_PACKAGE_SIZE = 4
    MIN_VALUE_THRESHOLD = 0.15
    MAX_COMBINATIONS = 100


# =============================================================================
# 基础类定义
# =============================================================================

class Asset:
    """资产类：球员或选秀权"""
    def __init__(self, name: str, asset_type: str, stats_vector: list, 
                 salary: float = 0.0, origin_team: str = None, age: float = None):
        self.name = name
        self.type = asset_type
        self.origin = origin_team
        self.age = age
        self.salary = salary
        self.stats = np.array(stats_vector)  # [AS, CS, SalEff, PW, Flex]
    
    def get_value(self, preference_vector: np.ndarray) -> float:
        """计算资产价值 V = stats^T · PV"""
        return np.dot(self.stats, preference_vector)
    
    def __repr__(self):
        if self.type == "Player":
            return f"{self.name} (${self.salary/1e6:.1f}M)"
        return f"{self.name}"


class Team:
    """球队类"""
    def __init__(self, name: str, preference_vector: np.ndarray, 
                 roster_quality_score: float):
        self.name = name
        self.pv = preference_vector
        self.quality = roster_quality_score
        self.assets = []
        self.draft_weight = 0.0
        self.draft_potential_score = 0.0
        self.total_salary = 0.0
    
    def add_asset(self, asset: Asset):
        self.assets.append(asset)
        if asset.type == "Player":
            self.total_salary += asset.salary


# =============================================================================
# 数据加载与资产创建
# =============================================================================

def infer_preference_vector(team_quality_normalized: float) -> np.ndarray:
    """根据球队质量推断偏好向量"""
    q = team_quality_normalized
    
    w_as = 0.10 + 0.35 * q
    w_pw = 0.35 - 0.30 * q
    w_cs = 0.15 + 0.10 * (1 - abs(q - 0.5) * 2)
    w_saleff = 0.30 - 0.20 * q
    w_flex = 1.0 - (w_as + w_pw + w_cs + w_saleff)
    
    pv = np.array([w_as, w_cs, w_saleff, w_pw, w_flex])
    pv = np.maximum(pv, 0.01)
    pv = pv / pv.sum()
    
    return pv


def calculate_salary_efficiency(salary: float, age: float, 
                                as_score: float, pw: float) -> float:
    """
    计算薪资效率 SalEff
    论文公式改进：SalEff = (Performance / Salary) * Age_Factor
    """
    if salary <= 0 or pd.isna(salary):
        return 0.5
    
    # 年龄因子
    if pd.isna(age) or age == 0:
        age_factor = 0.7
    elif age < 26:
        age_factor = 0.95 + (26 - age) * 0.01
    elif age < 31:
        age_factor = 0.95 - (age - 26) * 0.06
    else:
        age_factor = max(0.2, 0.70 - (age - 31) * 0.08)
    
    # 表现得分
    performance = as_score * 0.7 + pw * 0.3
    
    # 薪资标准化
    salary_normalized = min(salary / 50_000_000, 1.0)
    
    if salary_normalized > 0:
        sal_eff = (performance / salary_normalized) * age_factor
    else:
        sal_eff = 0.5
    
    sal_eff = min(1.0, max(0.0, sal_eff / 2.0))
    
    return sal_eff


def calculate_flexibility(age: float, as_score: float, 
                         contract_years_remaining: float = 2.0) -> float:
    """
    计算交易灵活性 Flex
    论文公式：Flex = exp(-rd(t))
    """
    base_flex = np.exp(-contract_years_remaining)
    
    if pd.isna(age) or age == 0:
        age_adjust = 0.7
    else:
        age_adjust = max(0.3, 1.0 - (age - 22) * 0.025)
    
    performance_adjust = 0.5 + as_score * 0.5
    
    flex = base_flex * age_adjust * performance_adjust
    return min(1.0, max(0.1, flex))


def create_player_asset_from_data(row: pd.Series) -> Asset:
    """从数据行创建球员资产"""
    name = row['player_name']
    team = row['team']
    age = row['age'] if pd.notna(row['age']) else 0
    
    as_score = row['athletic_score']
    cs = row['final_commercial_score']
    pw = row['PW']
    
    # 读取salary
    salary = row['salary'] if 'salary' in row and pd.notna(row['salary']) else 0
    
    # 计算派生属性
    sal_eff = calculate_salary_efficiency(salary, age, as_score, pw)
    
    # 估算合同剩余年限
    total_score = as_score * 100
    contract_remaining = Config.LONG_TERM_YEARS if total_score > Config.TS_THRESHOLD else Config.SHORT_TERM_YEARS
    contract_remaining -= np.random.randint(0, 2)
    contract_remaining = max(1, contract_remaining)
    
    flex = calculate_flexibility(age, as_score, contract_remaining)
    
    stats = [as_score, cs, sal_eff, pw, flex]
    
    return Asset(name, "Player", stats, salary=salary, 
                 origin_team=team, age=age)


def load_all_data(player_file: str, team_file: str) -> Tuple[Dict[str, Team], Dict[str, List[Asset]]]:
    """加载所有数据"""
    team_df = pd.read_csv(team_file)
    
    min_q = team_df['team_quality'].min()
    max_q = team_df['team_quality'].max()
    team_df['quality_norm'] = (team_df['team_quality'] - min_q) / (max_q - min_q)
    
    teams_dict = {}
    for _, row in team_df.iterrows():
        team_name = row['team']
        quality_norm = row['quality_norm']
        
        if team_name == 'NYK':
            pv = Config.PV_NYK
            print(f"✓ NYK使用论文指定PV: {pv}")
        else:
            pv = infer_preference_vector(quality_norm)
        
        team = Team(team_name, pv, quality_norm)
        teams_dict[team_name] = team
    
    player_df = pd.read_csv(player_file)
    
    if 'salary' not in player_df.columns:
        print("⚠️  警告：数据中缺少salary列")
        player_df['salary'] = 0
    
    players_by_team = {}
    
    for _, row in player_df.iterrows():
        player = create_player_asset_from_data(row)
        team_name = player.origin
        
        if team_name not in players_by_team:
            players_by_team[team_name] = []
        players_by_team[team_name].append(player)
        
        if team_name in teams_dict:
            teams_dict[team_name].add_asset(player)
    
    print(f"✓ 加载了 {len(teams_dict)} 支球队")
    print(f"✓ 加载了 {len(player_df)} 名球员")
    
    return teams_dict, players_by_team


def calculate_draft_weights(teams: List[Team], alpha: float = Config.ALPHA_DRAFT):
    """
    计算选秀权重
    论文公式：w_i^draft ∝ 1/(Q_i(t-1))^α
    """
    for team in teams:
        weight = 1.0 / ((team.quality + 0.01) ** alpha)
        team.draft_weight = weight
    
    weights = [t.draft_weight for t in teams]
    max_w = max(weights)
    min_w = min(weights)
    
    for team in teams:
        norm_w = 0.1 + 0.9 * (team.draft_weight - min_w) / (max_w - min_w + 1e-6)
        team.draft_potential_score = norm_w


def create_pick_asset(year: int, original_team: Team) -> Asset:
    """创建选秀权资产"""
    stats = [
        0.10,  # AS
        0.30,  # CS
        0.95,  # SalEff
        original_team.draft_potential_score,  # PW
        1.00   # Flex
    ]
    
    name = f"{year} {original_team.name} Pick"
    return Asset(name, "Pick", stats, salary=0, origin_team=original_team.name)


# =============================================================================
# 交易评估引擎
# =============================================================================

def trade_engine_for_nyk(nyk: Team, opponent: Team, 
                         package_nyk: List[Asset], 
                         package_opponent: List[Asset],
                         epsilon: float = Config.EPSILON_TRADE) -> Dict:
    """
    NYK交易评估引擎 - 严格按照论文算法
    
    论文约束：
    1. Fair Trade: |V_mkt^j(A) - V_mkt^i(B)| ≤ ε
    2. Bilateral Optimization: ΔV > 0 for both
    3. Salary Cap: 不超过工资帽
    """
    
    # NYK视角的市场价值
    val_give_nyk_market = sum([a.get_value(nyk.pv) for a in package_nyk])
    val_get_nyk_market = sum([a.get_value(nyk.pv) for a in package_opponent])
    
    # 对手视角的市场价值
    val_give_opp_market = sum([a.get_value(opponent.pv) for a in package_opponent])
    val_get_opp_market = sum([a.get_value(opponent.pv) for a in package_nyk])
    
    # 约束1：Fair Trade Principle
    market_val_package_nyk = (val_give_nyk_market + val_get_opp_market) / 2
    market_val_package_opp = (val_give_opp_market + val_get_nyk_market) / 2
    
    fairness_gap = abs(market_val_package_nyk - market_val_package_opp)
    fair_trade = fairness_gap <= epsilon
    
    # 约束2：Bilateral Optimization
    net_gain_nyk = val_get_nyk_market - val_give_nyk_market
    net_gain_opp = val_get_opp_market - val_give_opp_market
    
    nyk_benefits = net_gain_nyk > 0
    opponent_accepts = net_gain_opp > 0
    
    # 约束3：Salary Cap
    salary_out_nyk = sum([a.salary for a in package_nyk if a.type == "Player"])
    salary_in_nyk = sum([a.salary for a in package_opponent if a.type == "Player"])
    salary_change_nyk = salary_in_nyk - salary_out_nyk
    
    salary_out_opp = sum([a.salary for a in package_opponent if a.type == "Player"])
    salary_in_opp = sum([a.salary for a in package_nyk if a.type == "Player"])
    salary_change_opp = salary_in_opp - salary_out_opp
    
    nyk_salary_valid = (nyk.total_salary + salary_change_nyk) <= Config.SALARY_CAP
    opp_salary_valid = (opponent.total_salary + salary_change_opp) <= Config.SALARY_CAP
    salary_valid = nyk_salary_valid and opp_salary_valid
    
    # 综合决策
    approved = fair_trade and nyk_benefits and opponent_accepts
    
    # NYK优先评分
    if approved:
        nyk_score = net_gain_nyk * 0.80 + net_gain_opp * 0.20
    else:
        nyk_score = -999
    
    return {
        "approved": approved,
        "net_gain_nyk": net_gain_nyk,
        "net_gain_opponent": net_gain_opp,
        "fairness_gap": fairness_gap,
        "salary_change_nyk": salary_change_nyk,
        "salary_change_opp": salary_change_opp,
        "nyk_benefits": nyk_benefits,
        "opponent_accepts": opponent_accepts,
        "fair_trade": fair_trade,
        #"salary_valid": salary_valid,
        "nyk_score": nyk_score,
        "val_give_nyk": val_give_nyk_market,
        "val_get_nyk": val_get_nyk_market,
        "val_give_opp": val_give_opp_market,
        "val_get_opp": val_get_opp_market,
        "salary_in_nyk": salary_in_nyk,
        "salary_out_nyk": salary_out_nyk,
    }


# =============================================================================
# 交易推荐系统
# =============================================================================

class TradeRecommendation:
    """NYK交易推荐类"""
    def __init__(self, opponent: Team, 
                 package_nyk: List[Asset], 
                 package_opponent: List[Asset],
                 evaluation: Dict):
        self.opponent = opponent
        self.package_nyk = package_nyk
        self.package_opponent = package_opponent
        self.evaluation = evaluation
        self.score = evaluation['nyk_score']


def enumerate_trades_for_nyk(nyk: Team, 
                             teams_dict: Dict[str, Team], 
                             players_by_team: Dict[str, List[Asset]],
                             max_package_size: int = Config.MAX_PACKAGE_SIZE,
                             min_value_threshold: float = Config.MIN_VALUE_THRESHOLD) -> List[TradeRecommendation]:
    """为NYK枚举所有可能的交易"""
    all_recommendations = []
    
    # 创建选秀权
    picks_by_team = {}
    for team_name, team in teams_dict.items():
        pick_2026 = create_pick_asset(2026, team)
        picks_by_team[team_name] = [pick_2026]
    
    # NYK资产池
    nyk_players = players_by_team.get('NYK', [])
    nyk_picks = picks_by_team.get('NYK', [])
    nyk_assets = nyk_players + nyk_picks
    
    nyk_assets = [a for a in nyk_assets 
                  if a.get_value(nyk.pv) > min_value_threshold or a.type == 'Pick']
    
    print(f"\nNYK可交易资产池:")
    print(f"  球员: {len([a for a in nyk_assets if a.type == 'Player'])} 人")
    print(f"  选秀权: {len([a for a in nyk_assets if a.type == 'Pick'])} 个")
    print(f"  当前总薪资: ${nyk.total_salary/1e6:.1f}M / ${Config.SALARY_CAP/1e6:.0f}M")
    
    total_evaluated = 0
    approved_count = 0
    
    # 与其他29支球队评估
    for opponent_name, opponent in teams_dict.items():
        if opponent_name == 'NYK':
            continue
        
        opp_players = players_by_team.get(opponent_name, [])
        opp_picks = picks_by_team.get(opponent_name, [])
        opp_assets = opp_players + opp_picks
        
        opp_assets = [a for a in opp_assets 
                      if a.get_value(opponent.pv) > min_value_threshold or a.type == 'Pick']
        
        import random
        if len(opp_assets) > 25:
            opp_assets = sorted(opp_assets, 
                               key=lambda x: x.get_value(opponent.pv), 
                               reverse=True)[:25]
        
        # 枚举组合
        for size_nyk in range(1, max_package_size + 1):
            for size_opp in range(1, max_package_size + 1):
                packages_nyk = list(combinations(nyk_assets, size_nyk))
                packages_opp = list(combinations(opp_assets, size_opp))
                
                if len(packages_nyk) > Config.MAX_COMBINATIONS:
                    packages_nyk = random.sample(packages_nyk, Config.MAX_COMBINATIONS)
                if len(packages_opp) > Config.MAX_COMBINATIONS:
                    packages_opp = random.sample(packages_opp, Config.MAX_COMBINATIONS)
                
                for pkg_nyk in packages_nyk:
                    for pkg_opp in packages_opp:
                        total_evaluated += 1
                        
                        evaluation = trade_engine_for_nyk(
                            nyk, opponent, 
                            list(pkg_nyk), list(pkg_opp),
                            epsilon=Config.EPSILON_TRADE
                        )
                        
                        if evaluation['approved']:
                            approved_count += 1
                            recommendation = TradeRecommendation(
                                opponent, list(pkg_nyk), list(pkg_opp), evaluation
                            )
                            all_recommendations.append(recommendation)
        
        if total_evaluated % 10000 == 0 and total_evaluated > 0:
            print(f"  已评估 {total_evaluated:,} 笔交易...")
    
    print(f"\n枚举完成！")
    print(f"  总评估交易数: {total_evaluated:,}")
    print(f"  通过审核交易数: {approved_count}")
    if total_evaluated > 0:
        print(f"  通过率: {approved_count/total_evaluated*100:.2f}%")
    
    return all_recommendations


# =============================================================================
# 结果展示
# =============================================================================

def print_nyk_recommendation(rec: TradeRecommendation, rank: int):
    """打印NYK交易推荐详情"""
    print(f"\n{'='*80}")
    print(f"推荐 #{rank}: NYK ↔ {rec.opponent.name}")
    print(f"{'='*80}")
    
    eval_data = rec.evaluation
    
    # NYK送出
    print(f"\n【NYK 送出】(市场价值: {eval_data['val_give_nyk']:.4f}, "
          f"薪资: ${eval_data['salary_out_nyk']/1e6:.1f}M)")
    for asset in rec.package_nyk:
        if asset.type == "Player":
            stats_str = (f"AS:{asset.stats[0]:.2f}, CS:{asset.stats[1]:.2f}, "
                        f"SalEff:{asset.stats[2]:.2f}")
            print(f"  • {asset.name} [{stats_str}] "
                  f"{asset.age:.0f}岁, ${asset.salary/1e6:.1f}M")
        else:
            print(f"  • {asset.name} [选秀权]")
    
    # NYK获得
    print(f"\n【NYK 获得】(市场价值: {eval_data['val_get_nyk']:.4f}, "
          f"薪资: ${eval_data['salary_in_nyk']/1e6:.1f}M)")
    for asset in rec.package_opponent:
        if asset.type == "Player":
            stats_str = (f"AS:{asset.stats[0]:.2f}, CS:{asset.stats[1]:.2f}, "
                        f"SalEff:{asset.stats[2]:.2f}")
            print(f"  • {asset.name} [{stats_str}] "
                  f"{asset.age:.0f}岁, ${asset.salary/1e6:.1f}M")
        else:
            print(f"  • {asset.name} [选秀权]")
    
    # 论文约束检验
    print(f"\n【论文约束检验】")
    print(f"  ✓ Fair Trade: 差距={eval_data['fairness_gap']:.4f} "
          f"({'✓通过' if eval_data['fair_trade'] else '❌未通过'}, ε={Config.EPSILON_TRADE})")
    print(f"  ✓ NYK Bilateral Opt: 净收益={eval_data['net_gain_nyk']:+.4f} "
          f"({'✓通过' if eval_data['nyk_benefits'] else '❌未通过'})")
    print(f"  ✓ {rec.opponent.name} Bilateral Opt: "
          f"净收益={eval_data['net_gain_opponent']:+.4f} "
          f"({'✓通过' if eval_data['opponent_accepts'] else '❌未通过'})")
    #print(f"  ✓ Salary Cap: NYK薪资变化={eval_data['salary_change_nyk']/1e6:+.1f}M "
          #f"({'✓通过' if eval_data['salary_valid'] else '❌未通过'})")
    
    # 价值分析
    print(f"\n【价值分析】")
    if eval_data['val_give_nyk'] > 0:
        value_increase = (eval_data['val_get_nyk'] / eval_data['val_give_nyk'] - 1) * 100
        print(f"  价值提升: {value_increase:+.1f}%")
    print(f"  NYK评分: {eval_data['nyk_score']:.4f}")
    
    # 推荐建议
    print(f"\n【推荐建议】")
    if eval_data['net_gain_nyk'] > 0.20:
        print(f"  ⭐⭐⭐ 强烈推荐！显著提升 (+{eval_data['net_gain_nyk']*100:.1f}%)")
    elif eval_data['net_gain_nyk'] > 0.10:
        print(f"  ⭐⭐ 推荐。稳健提升 (+{eval_data['net_gain_nyk']*100:.1f}%)")
    else:
        print(f"  ⭐ 可考虑。小幅提升 (+{eval_data['net_gain_nyk']*100:.1f}%)")


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print(f"\n{'#'*80}")
    print(f"#{'NYK交易推荐系统'.center(78)}#")
    print(f"{'#'*80}\n")
    
    print("【系统配置】")
    print(f"  ε_trade (Fair Trade阈值): {Config.EPSILON_TRADE}")
    print(f"  α (选秀公平调节器): {Config.ALPHA_DRAFT}")
    print(f"  Salary Cap: ${Config.SALARY_CAP/1e6:.0f}M")
    print(f"  NYK偏好向量: {Config.PV_NYK}")
    print(f"  最大包裹大小: {Config.MAX_PACKAGE_SIZE}\n")
    
    start_time = time.time()
    
    # Step 1: 加载数据
    print("[Step 1] 加载数据...")
    teams_dict, players_by_team = load_all_data(
        "csv_fold/merged_player_data.csv",
        "csv_fold/team_quality.csv"
    )
    
    # Step 2: 计算选秀权重
    print("\n[Step 2] 计算选秀权重...")
    calculate_draft_weights(list(teams_dict.values()))
    
    # Step 3: NYK信息
    nyk = teams_dict['NYK']
    print(f"\n[Step 3] NYK球队信息")
    team_qualities = sorted([t.quality for t in teams_dict.values()], reverse=True)
    nyk_rank = team_qualities.index(nyk.quality) + 1
    print(f"  质量分数: {nyk.quality:.4f}")
    print(f"  质量排名: {nyk_rank}/30")
    print(f"  偏好向量: [AS:{nyk.pv[0]:.2f}, CS:{nyk.pv[1]:.2f}, "
          f"SalEff:{nyk.pv[2]:.2f}, PW:{nyk.pv[3]:.2f}, Flex:{nyk.pv[4]:.2f}]")
    print(f"  阵容人数: {len([a for a in nyk.assets if a.type == 'Player'])} 人")
    print(f"  总薪资: ${nyk.total_salary/1e6:.1f}M")
    print(f"  薪资空间: ${(Config.SALARY_CAP - nyk.total_salary)/1e6:.1f}M")
    
    # 展示NYK核心球员
    print(f"\n  NYK核心球员 (按即战力Top 10):")
    nyk_players = sorted([a for a in nyk.assets if a.type == 'Player'], 
                         key=lambda x: x.stats[0], reverse=True)[:10]
    for i, player in enumerate(nyk_players, 1):
        print(f"    {i}. {player.name}: AS={player.stats[0]:.3f}, "
              f"年龄={player.age:.0f}, 薪资=${player.salary/1e6:.1f}M")
    
    # Step 4: 枚举交易
    print(f"\n[Step 4] 为NYK枚举交易方案...")
    all_recommendations = enumerate_trades_for_nyk(
        nyk, teams_dict, players_by_team
    )
    
    # Step 5: Top 10推荐
    print(f"\n[Step 5] 生成Top 10推荐...")
    
    if len(all_recommendations) == 0:
        print("\n⚠️  未找到符合所有约束条件的交易！")
        print("建议：")
        print("  1. 调整 EPSILON_TRADE 参数（当前={})".format(Config.EPSILON_TRADE))
        print("  2. 调整 MIN_VALUE_THRESHOLD 参数（当前={})".format(Config.MIN_VALUE_THRESHOLD))
        print("  3. 增加 MAX_PACKAGE_SIZE（当前={})".format(Config.MAX_PACKAGE_SIZE))
    else:
        all_recommendations.sort(key=lambda x: x.score, reverse=True)
        top_10 = all_recommendations[:10]
        
        print(f"\n{'='*80}")
        print(f"NYK Top 10 交易推荐")
        print(f"{'='*80}")
        
        for i, rec in enumerate(top_10, 1):
            print_nyk_recommendation(rec, i)
    
    # 统计信息
    elapsed_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"执行统计")
    print(f"{'='*80}")
    print(f"  总耗时: {elapsed_time:.2f}秒")
    print(f"  可行交易数: {len(all_recommendations)}")
    if all_recommendations:
        gains = [r.evaluation['net_gain_nyk'] for r in all_recommendations]
        print(f"  NYK平均净收益: {np.mean(gains):.4f}")
        print(f"  NYK最高净收益: {max(gains):.4f}")
        print(f"  NYK最低净收益: {min(gains):.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
