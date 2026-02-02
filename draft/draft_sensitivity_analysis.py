"""
选秀策略敏感性分析
分析新秀属性对NYK的影响
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 配置参数
# =============================================================================

class Config:
    """系统配置参数"""
    ALPHA_DRAFT = 2.0  # α: 选秀公平性调节器
    PV_NYK = np.array([0.40, 0.18, 0.13, 0.10, 0.20])
    # [AS, CS, SalEff, PW, Flex]


# =============================================================================
# 基础类
# =============================================================================

class Team:
    """球队类"""
    def __init__(self, name: str, preference_vector: np.ndarray, 
                 roster_quality_score: float):
        self.name = name
        self.pv = preference_vector
        self.quality = roster_quality_score
        self.draft_weight = 0.0
        self.draft_potential_score = 0.0


class Asset:
    """资产类"""
    def __init__(self, name: str, asset_type: str, stats_vector: list):
        self.name = name
        self.type = asset_type
        self.stats = np.array(stats_vector)
    
    def get_value(self, preference_vector: np.ndarray) -> float:
        return np.dot(self.stats, preference_vector)


# =============================================================================
# 数据加载
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


def load_teams(team_file: str) -> Dict[str, Team]:
    """加载球队数据"""
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
        else:
            pv = infer_preference_vector(quality_norm)
        
        team = Team(team_name, pv, quality_norm)
        teams_dict[team_name] = team
    
    print(f"✓ 加载了 {len(teams_dict)} 支球队")
    
    return teams_dict


# =============================================================================
# 选秀权重计算
# =============================================================================

def calculate_draft_weights(teams: List[Team], alpha: float):
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


# =============================================================================
# 敏感性分析器
# =============================================================================

class DraftSensitivityAnalyzer:
    """选秀策略敏感性分析器"""
    
    def __init__(self, teams_dict: Dict[str, Team]):
        self.teams = list(teams_dict.values())
        self.nyk = teams_dict.get('NYK')
    
    def analyze_alpha_sensitivity(self, alpha_range: np.ndarray) -> Dict:
        """
        分析α（选秀公平调节器）的敏感性
        
        论文公式：w_i^draft ∝ 1/(Q_i(t-1))^α
        """
        results = {
            'alpha_values': [],
            'nyk_draft_weight': [],
            'nyk_draft_potential': [],
            'gini_coefficient': []
        }
        
        for alpha in alpha_range:
            # 计算选秀权重
            calculate_draft_weights(self.teams, alpha=alpha)
            
            results['alpha_values'].append(alpha)
            results['nyk_draft_weight'].append(self.nyk.draft_weight)
            results['nyk_draft_potential'].append(self.nyk.draft_potential_score)
            
            # 计算基尼系数
            weights = [t.draft_weight for t in self.teams]
            gini = self._calculate_gini(weights)
            results['gini_coefficient'].append(gini)
        
        return results
    
    def analyze_rookie_attributes(self, attribute_ranges: Dict[str, np.ndarray]) -> Dict:
        """
        分析新秀属性对NYK价值的敏感性
        
        属性：AS, CS, SalEff, PW, Flex
        """
        results = {}
        
        # 基准新秀属性
        base_stats = [0.50, 0.40, 0.90, 0.70, 0.85]
        
        for attr_name, attr_values in attribute_ranges.items():
            attr_index = ['AS', 'CS', 'SalEff', 'PW', 'Flex'].index(attr_name)
            
            values = []
            for attr_val in attr_values:
                stats = base_stats.copy()
                stats[attr_index] = attr_val
                
                # 计算对NYK的价值
                value_to_nyk = np.dot(stats, self.nyk.pv)
                values.append(value_to_nyk)
            
            results[attr_name] = {
                'attribute_values': attr_values,
                'nyk_values': values
            }
        
        return results
    
    def analyze_draft_position_value(self, num_picks: int = 60) -> Dict:
        """
        分析选秀顺位价值
        
        论文中 ΔTQ_i^D(t) = Σ w_j^draft · n_{i,j}(t)
        """
        results = {
            'pick_positions': [],
            'expected_values': [],
            'value_to_nyk': []
        }
        
        # 按质量排序球队（模拟选秀顺位）
        sorted_teams = sorted(self.teams, key=lambda t: t.quality)
        
        for i in range(min(num_picks, len(sorted_teams) * 2)):
            team_idx = i % len(sorted_teams)
            originating_team = sorted_teams[team_idx]
            
            # 创建选秀权
            pick_stats = [
                0.10,  # AS
                0.30,  # CS
                0.95,  # SalEff
                originating_team.draft_potential_score,  # PW
                1.00   # Flex
            ]
            
            pick = Asset(f"Pick {i+1}", "Pick", pick_stats)
            
            results['pick_positions'].append(i + 1)
            results['expected_values'].append(pick_stats[3])  # PW
            results['value_to_nyk'].append(pick.get_value(self.nyk.pv))
        
        return results
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    def plot_sensitivity_analysis(self, save_path: str = "draft_sensitivity_analysis.pdf"):
        """绘制敏感性分析图表"""
        # 只绘制Gini系数图，尺寸3.4×2.4
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        
        # α敏感性分析
        alpha_range = np.linspace(0.5, 5.0, 50)
        alpha_results = self.analyze_alpha_sensitivity(alpha_range)
        
        # 绘制基尼系数变化
        ax.plot(alpha_results['alpha_values'], 
                alpha_results['gini_coefficient'], 
                'g-', linewidth=2)
        ax.axvline(x=Config.ALPHA_DRAFT, color='r', linestyle='--',
                  label=f'Current α={Config.ALPHA_DRAFT}')
        ax.set_xlabel('α (Fairness Regulator)', fontsize=10)
        ax.set_ylabel('Gini Coefficient', fontsize=10)
        ax.set_title('League Draft Equality', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 敏感性分析图表已保存到: {save_path}")
        
        return fig
    
    def generate_sensitivity_report(self) -> str:
        """生成敏感性分析文字报告"""
        report = []
        report.append("\n" + "="*80)
        report.append("选秀策略敏感性分析报告 - NYK视角")
        report.append("="*80)
        
        # α敏感性
        alpha_range = np.linspace(0.5, 5.0, 20)
        alpha_results = self.analyze_alpha_sensitivity(alpha_range)
        
        report.append("\n【1. α参数敏感性分析】")
        report.append(f"当前α = {Config.ALPHA_DRAFT}")
        report.append(f"NYK当前选秀潜力得分: {self.nyk.draft_potential_score:.4f}")
        
        idx_current = np.argmin(np.abs(alpha_range - Config.ALPHA_DRAFT))
        current_potential = alpha_results['nyk_draft_potential'][idx_current]
        
        # 找最优α
        idx_best = np.argmax(alpha_results['nyk_draft_potential'])
        best_alpha = alpha_results['alpha_values'][idx_best]
        best_potential = alpha_results['nyk_draft_potential'][idx_best]
        
        report.append(f"\n最优α配置: {best_alpha:.2f}")
        report.append(f"  在该α下，NYK选秀潜力 = {best_potential:.4f}")
        report.append(f"  相比当前提升: {((best_potential/current_potential - 1) * 100):.1f}%")
        
        # 新秀属性敏感性
        report.append("\n【2. 新秀属性敏感性排名】")
        
        attribute_ranges = {
            'AS': np.linspace(0.3, 0.9, 10),
            'CS': np.linspace(0.2, 0.8, 10),
            'PW': np.linspace(0.4, 1.0, 10),
            'SalEff': np.linspace(0.6, 1.0, 10),
            'Flex': np.linspace(0.5, 1.0, 10),
        }
        
        attr_results = self.analyze_rookie_attributes(attribute_ranges)
        
        sensitivities = {}
        for attr_name, data in attr_results.items():
            values = np.array(data['nyk_values'])
            # 使用标准差/均值作为敏感度指标
            sensitivity = np.std(values) / np.mean(values)
            sensitivities[attr_name] = sensitivity
        
        sorted_attrs = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (attr, sens) in enumerate(sorted_attrs, 1):
            pv_weight = self.nyk.pv[['AS', 'CS', 'SalEff', 'PW', 'Flex'].index(attr)]
            report.append(f"  {rank}. {attr}: 敏感度={sens:.4f}, NYK偏好权重={pv_weight:.2f}")
        
        # 选秀顺位分析
        report.append("\n【3. 关键选秀顺位价值（NYK视角）】")
        draft_results = self.analyze_draft_position_value(num_picks=30)
        
        top_5_picks = draft_results['pick_positions'][:5]
        top_5_values = draft_results['value_to_nyk'][:5]
        
        for pos, val in zip(top_5_picks, top_5_values):
            report.append(f"  第{pos}顺位: 对NYK价值 = {val:.4f}")
        
        # NYK当前状态分析
        report.append("\n【4. NYK当前状态】")
        report.append(f"  球队质量分数: {self.nyk.quality:.4f}")
        
        all_qualities = sorted([t.quality for t in self.teams], reverse=True)
        nyk_rank = all_qualities.index(self.nyk.quality) + 1
        report.append(f"  联盟排名: {nyk_rank}/30")
        
        # 重新计算一次以确保最新数据
        calculate_draft_weights(self.teams, alpha=Config.ALPHA_DRAFT)
        report.append(f"  选秀权重: {self.nyk.draft_weight:.4f}")
        report.append(f"  选秀潜力: {self.nyk.draft_potential_score:.4f}")
        
        # 策略建议
        report.append("\n【5. NYK选秀策略建议】")
        report.append(f"  基于NYK的偏好向量 {self.nyk.pv}:")
        report.append(f"  1. 最看重属性: AS (权重={self.nyk.pv[0]:.2f})")
        report.append(f"  2. 次看重属性: CS (权重={self.nyk.pv[1]:.2f})")
        report.append(f"  3. 建议目标: 即战力强、商业价值高的新秀")
        


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print(f"\n{'#'*80}")
    print(f"#{'选秀策略敏感性分析系统'.center(78)}#")
    print(f"{'#'*80}\n")
    
    print("【系统配置】")
    print(f"  α (选秀公平调节器): {Config.ALPHA_DRAFT}")
    print(f"  NYK偏好向量: {Config.PV_NYK}")
    print()
    
    # Step 1: 加载数据
    print("[Step 1] 加载球队数据...")
    teams_dict = load_teams("csv_fold/team_quality.csv")
    
    # Step 2: 计算选秀权重
    print("\n[Step 2] 计算选秀权重...")
    calculate_draft_weights(list(teams_dict.values()), alpha=Config.ALPHA_DRAFT)
    
    # Step 3: 创建分析器
    print("\n[Step 3] 初始化敏感性分析器...")
    analyzer = DraftSensitivityAnalyzer(teams_dict)
    
    # Step 4: 生成图表
    print("\n[Step 4] 生成敏感性分析图表...")
    analyzer.plot_sensitivity_analysis("draft_sensitivity_analysis.pdf")
    
    # Step 5: 生成报告
    print("\n[Step 5] 生成分析报告...")
    report = analyzer.generate_sensitivity_report()
    print(report)
    
    # 保存报告到文件
    report_file = "draft_sensitivity_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ 分析报告已保存到: {report_file}")
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
