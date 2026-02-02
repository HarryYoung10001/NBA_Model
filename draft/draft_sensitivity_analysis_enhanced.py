"""
选秀策略敏感性分析 - 增强版
对论文中的关键公式进行全面敏感性分析
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

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
# 增强版敏感性分析器
# =============================================================================

class EnhancedDraftSensitivityAnalyzer:
    """增强版选秀策略敏感性分析器"""
    
    def __init__(self, teams_dict: Dict[str, Team]):
        self.teams = list(teams_dict.values())
        self.nyk = teams_dict.get('NYK')
        self.sensitivity_results = {}
    
    # =========================================================================
    # 1. α参数敏感性分析
    # =========================================================================
    
    def analyze_alpha_sensitivity(self, alpha_range: np.ndarray) -> Dict:
        """
        分析α（选秀公平调节器）的敏感性
        
        论文公式：w_i^draft ∝ 1/(Q_i(t-1))^α
        
        敏感性指标：
        1. NYK选秀权重变化率
        2. 联盟基尼系数变化
        3. 弱队受益程度
        """
        results = {
            'alpha_values': [],
            'nyk_draft_weight': [],
            'nyk_draft_potential': [],
            'gini_coefficient': [],
            'bottom_5_avg_weight': [],  # 最弱5队平均权重
            'top_5_avg_weight': []      # 最强5队平均权重
        }
        
        for alpha in alpha_range:
            calculate_draft_weights(self.teams, alpha=alpha)
            
            results['alpha_values'].append(alpha)
            results['nyk_draft_weight'].append(self.nyk.draft_weight)
            results['nyk_draft_potential'].append(self.nyk.draft_potential_score)
            
            # 基尼系数
            weights = [t.draft_weight for t in self.teams]
            gini = self._calculate_gini(weights)
            results['gini_coefficient'].append(gini)
            
            # 弱队和强队的平均权重
            sorted_teams = sorted(self.teams, key=lambda t: t.quality)
            bottom_5_weights = [t.draft_weight for t in sorted_teams[:5]]
            top_5_weights = [t.draft_weight for t in sorted_teams[-5:]]
            
            results['bottom_5_avg_weight'].append(np.mean(bottom_5_weights))
            results['top_5_avg_weight'].append(np.mean(top_5_weights))
        
        # 计算敏感度指标
        alpha_sensitivity = self._calculate_sensitivity_index(
            alpha_range, 
            results['nyk_draft_potential']
        )
        
        results['sensitivity_index'] = alpha_sensitivity
        
        self.sensitivity_results['alpha'] = results
        return results
    
    # =========================================================================
    # 2. 偏好向量敏感性分析
    # =========================================================================
    
    def analyze_preference_vector_sensitivity(self, pv_perturbation: float = 0.1) -> Dict:
        """
        分析偏好向量PV各分量的敏感性
        
        论文公式：V_mkt^i(A) = V_A^T · PV_i
        
        方法：对每个分量±perturbation，观察对选秀价值的影响
        """
        results = {}
        pv_names = ['AS', 'CS', 'SalEff', 'PW', 'Flex']
        
        # 基准新秀属性
        base_rookie_stats = [0.50, 0.40, 0.90, 0.70, 0.85]
        base_value = np.dot(base_rookie_stats, self.nyk.pv)
        
        for i, pv_name in enumerate(pv_names):
            perturbed_values = []
            perturbation_range = np.linspace(-pv_perturbation, pv_perturbation, 21)
            
            for delta in perturbation_range:
                # 扰动PV
                perturbed_pv = self.nyk.pv.copy()
                perturbed_pv[i] += delta
                
                # 重新归一化
                if perturbed_pv[i] < 0:
                    perturbed_pv[i] = 0
                perturbed_pv = perturbed_pv / perturbed_pv.sum()
                
                # 计算新秀价值
                new_value = np.dot(base_rookie_stats, perturbed_pv)
                perturbed_values.append(new_value)
            
            # 计算敏感度
            sensitivity = np.std(perturbed_values) / np.mean(perturbed_values)
            
            results[pv_name] = {
                'perturbations': perturbation_range,
                'values': perturbed_values,
                'sensitivity': sensitivity,
                'base_weight': self.nyk.pv[i]
            }
        
        self.sensitivity_results['preference_vector'] = results
        return results
    
    # =========================================================================
    # 3. 新秀属性敏感性分析
    # =========================================================================
    
    def analyze_rookie_attributes(self, attribute_ranges: Dict[str, np.ndarray]) -> Dict:
        """
        分析新秀属性对NYK价值的敏感性
        
        论文公式：V_A = [AS_A, CS_A, SalEff_A, PW_A, Flex_A]^T
        """
        results = {}
        
        # 基准新秀属性
        base_stats = [0.50, 0.40, 0.90, 0.70, 0.85]
        
        for attr_name, attr_values in attribute_ranges.items():
            attr_index = ['AS', 'CS', 'SalEff', 'PW', 'Flex'].index(attr_name)
            
            values = []
            marginal_values = []
            
            for i, attr_val in enumerate(attr_values):
                stats = base_stats.copy()
                stats[attr_index] = attr_val
                
                # 计算对NYK的价值
                value_to_nyk = np.dot(stats, self.nyk.pv)
                values.append(value_to_nyk)
                
                # 计算边际价值
                if i > 0:
                    marginal = (values[i] - values[i-1]) / (attr_values[i] - attr_values[i-1])
                    marginal_values.append(marginal)
            
            # 计算弹性系数
            elasticity = self._calculate_elasticity(attr_values, values)
            
            results[attr_name] = {
                'attribute_values': attr_values,
                'nyk_values': values,
                'marginal_values': marginal_values,
                'elasticity': elasticity,
                'pv_weight': self.nyk.pv[attr_index]
            }
        
        self.sensitivity_results['rookie_attributes'] = results
        return results
    
    # =========================================================================
    # 4. 合同灵活性敏感性分析
    # =========================================================================
    
    def analyze_flexibility_sensitivity(self) -> Dict:
        """
        分析合同灵活性公式的敏感性
        
        论文公式：
        Flex_A = e^(-rd(t))
        rd(t) = T_ct - (t - t_ct)
        
        分析不同剩余合同年限对球员价值的影响
        """
        results = {
            'remaining_years': [],
            'flexibility_values': [],
            'total_values': [],
            'flexibility_contribution': []
        }
        
        # 基准球员属性（不含Flex）
        base_stats = [0.60, 0.50, 0.85, 0.65, 0.0]  # 最后一项是Flex，待计算
        
        # 剩余合同年限范围：0.1到4年
        rd_range = np.linspace(0.1, 4.0, 40)
        
        for rd in rd_range:
            # 计算Flex
            flex = np.exp(-rd)
            
            # 完整属性向量
            full_stats = base_stats.copy()
            full_stats[4] = flex
            
            # 计算总价值
            total_value = np.dot(full_stats, self.nyk.pv)
            
            # Flex的贡献
            flex_contribution = flex * self.nyk.pv[4]
            
            results['remaining_years'].append(rd)
            results['flexibility_values'].append(flex)
            results['total_values'].append(total_value)
            results['flexibility_contribution'].append(flex_contribution)
        
        # 计算敏感度
        sensitivity = np.std(results['total_values']) / np.mean(results['total_values'])
        results['sensitivity'] = sensitivity
        
        self.sensitivity_results['flexibility'] = results
        return results
    
    # =========================================================================
    # 5. 选秀顺位价值敏感性
    # =========================================================================
    
    def analyze_draft_position_value(self, num_picks: int = 60) -> Dict:
        """
        分析选秀顺位价值
        
        论文公式：ΔTQ_i^D(t) = Σ w_j^draft · n_{i,j}(t)
        """
        results = {
            'pick_positions': [],
            'expected_values': [],
            'value_to_nyk': [],
            'cumulative_value': []
        }
        
        # 按质量排序球队（模拟选秀顺位）
        sorted_teams = sorted(self.teams, key=lambda t: t.quality)
        
        cumulative = 0
        for i in range(min(num_picks, len(sorted_teams) * 2)):
            team_idx = i % len(sorted_teams)
            originating_team = sorted_teams[team_idx]
            
            # 创建选秀权属性
            pick_stats = [
                0.10,  # AS
                0.30,  # CS
                0.95,  # SalEff
                originating_team.draft_potential_score,  # PW
                1.00   # Flex
            ]
            
            pick = Asset(f"Pick {i+1}", "Pick", pick_stats)
            value_to_nyk = pick.get_value(self.nyk.pv)
            cumulative += value_to_nyk
            
            results['pick_positions'].append(i + 1)
            results['expected_values'].append(pick_stats[3])  # PW
            results['value_to_nyk'].append(value_to_nyk)
            results['cumulative_value'].append(cumulative)
        
        self.sensitivity_results['draft_position'] = results
        return results
    
    # =========================================================================
    # 6. 交叉敏感性分析
    # =========================================================================
    
    def analyze_cross_sensitivity_alpha_pv(self) -> Dict:
        """
        分析α与PV权重的交互影响
        
        问题：当α变化时，不同PV配置下NYK的选秀收益如何变化？
        """
        alpha_range = np.linspace(0.5, 5.0, 20)
        pv_scenarios = {
            '当前配置': self.nyk.pv,
            '重AS轻PW': np.array([0.50, 0.15, 0.10, 0.05, 0.20]),
            '均衡配置': np.array([0.20, 0.20, 0.20, 0.20, 0.20]),
            '重PW轻AS': np.array([0.15, 0.15, 0.10, 0.40, 0.20])
        }
        
        results = {
            'alpha_values': alpha_range.tolist(),
            'scenarios': {}
        }
        
        for scenario_name, pv in pv_scenarios.items():
            scenario_values = []
            
            for alpha in alpha_range:
                calculate_draft_weights(self.teams, alpha=alpha)
                
                # 计算典型新秀对该PV配置的价值
                rookie_stats = [0.50, 0.40, 0.90, 0.70, 0.85]
                value = np.dot(rookie_stats, pv)
                
                # 加权NYK的选秀潜力
                weighted_value = value * self.nyk.draft_potential_score
                scenario_values.append(weighted_value)
            
            results['scenarios'][scenario_name] = scenario_values
        
        self.sensitivity_results['cross_alpha_pv'] = results
        return results
    
    # =========================================================================
    # 辅助函数
    # =========================================================================
    
    def _calculate_gini(self, values: List[float]) -> float:
        """计算基尼系数"""
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n
    
    def _calculate_sensitivity_index(self, x: np.ndarray, y: List[float]) -> float:
        """
        计算敏感度指数
        定义为：标准差与均值之比（变异系数）
        """
        return np.std(y) / np.mean(y)
    
    def _calculate_elasticity(self, x: np.ndarray, y: List[float]) -> float:
        """
        计算弹性系数
        定义为：(Δy/y) / (Δx/x) 的平均值
        """
        elasticities = []
        for i in range(1, len(x)):
            dx = (x[i] - x[i-1]) / x[i-1]
            dy = (y[i] - y[i-1]) / y[i-1]
            if abs(dx) > 1e-6:
                elasticities.append(dy / dx)
        return np.mean(elasticities) if elasticities else 0
    
    # =========================================================================
    # 可视化
    # =========================================================================
    
    def plot_comprehensive_analysis(self, save_path: str = "sensitivity_analysis.pdf"):
        """生成综合敏感性分析图表"""
        
        # 运行所有分析
        print("\n执行综合敏感性分析...")
        
        alpha_range = np.linspace(0.5, 5.0, 50)
        alpha_results = self.analyze_alpha_sensitivity(alpha_range)
        
        pv_results = self.analyze_preference_vector_sensitivity(pv_perturbation=0.1)
        
        attribute_ranges = {
            'AS': np.linspace(0.3, 0.9, 20),
            'CS': np.linspace(0.2, 0.8, 20),
            'PW': np.linspace(0.4, 1.0, 20),
            'SalEff': np.linspace(0.6, 1.0, 20),
            'Flex': np.linspace(0.5, 1.0, 20),
        }
        rookie_results = self.analyze_rookie_attributes(attribute_ranges)
        
        flex_results = self.analyze_flexibility_sensitivity()
        draft_results = self.analyze_draft_position_value(num_picks=60)
        cross_results = self.analyze_cross_sensitivity_alpha_pv()
        
        # 创建图表
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. α敏感性 - 基尼系数
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(alpha_results['alpha_values'], alpha_results['gini_coefficient'], 
                'b-', linewidth=2, label='基尼系数')
        ax1.axvline(x=Config.ALPHA_DRAFT, color='r', linestyle='--', 
                   label=f'当前 α={Config.ALPHA_DRAFT}')
        ax1.set_xlabel('α (公平性调节器)', fontsize=10)
        ax1.set_ylabel('基尼系数', fontsize=10)
        ax1.set_title('(1) α参数对联盟公平性的影响', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)
        
        # 2. α敏感性 - 强弱队对比
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(alpha_results['alpha_values'], alpha_results['bottom_5_avg_weight'], 
                'g-', linewidth=2, label='最弱5队')
        ax2.plot(alpha_results['alpha_values'], alpha_results['top_5_avg_weight'], 
                'r-', linewidth=2, label='最强5队')
        ax2.axvline(x=Config.ALPHA_DRAFT, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('α (公平性调节器)', fontsize=10)
        ax2.set_ylabel('平均选秀权重', fontsize=10)
        ax2.set_title('(2) α参数对强弱队选秀权重的影响', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=8)
        
        # 3. 偏好向量敏感性
        ax3 = fig.add_subplot(gs[0, 2])
        pv_names = list(pv_results.keys())
        sensitivities = [pv_results[name]['sensitivity'] for name in pv_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(pv_names)))
        bars = ax3.barh(pv_names, sensitivities, color=colors)
        ax3.set_xlabel('敏感度指数', fontsize=10)
        ax3.set_title('(3) 偏好向量各分量敏感性', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. 新秀属性敏感性
        ax4 = fig.add_subplot(gs[1, 0])
        for attr_name in ['AS', 'CS', 'PW']:
            data = rookie_results[attr_name]
            ax4.plot(data['attribute_values'], data['nyk_values'], 
                    linewidth=2, label=f"{attr_name} (弹性={data['elasticity']:.2f})")
        ax4.set_xlabel('属性值', fontsize=10)
        ax4.set_ylabel('对NYK的价值', fontsize=10)
        ax4.set_title('(4) 新秀关键属性敏感性', fontsize=11, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=8)
        
        # 5. 合同灵活性敏感性
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(flex_results['remaining_years'], flex_results['flexibility_values'], 
                'purple', linewidth=2, label='灵活性值')
        ax5_twin = ax5.twinx()
        ax5_twin.plot(flex_results['remaining_years'], flex_results['flexibility_contribution'], 
                     'orange', linewidth=2, linestyle='--', label='对总价值的贡献')
        ax5.set_xlabel('剩余合同年限', fontsize=10)
        ax5.set_ylabel('灵活性 Flex = e^(-rd)', fontsize=10, color='purple')
        ax5_twin.set_ylabel('对总价值贡献', fontsize=10, color='orange')
        ax5.set_title('(5) 合同灵活性公式敏感性', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='y', labelcolor='purple')
        ax5_twin.tick_params(axis='y', labelcolor='orange')
        
        # 6. 选秀顺位价值
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(draft_results['pick_positions'][:30], draft_results['value_to_nyk'][:30], 
                'b-', linewidth=2, marker='o', markersize=3)
        ax6.set_xlabel('选秀顺位', fontsize=10)
        ax6.set_ylabel('对NYK的价值', fontsize=10)
        ax6.set_title('(6) 前30顺位选秀权价值', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.invert_xaxis()
        
        # 7. α与PV交互影响
        ax7 = fig.add_subplot(gs[2, :2])
        for scenario_name, values in cross_results['scenarios'].items():
            ax7.plot(cross_results['alpha_values'], values, 
                    linewidth=2, marker='o', markersize=4, label=scenario_name)
        ax7.axvline(x=Config.ALPHA_DRAFT, color='gray', linestyle='--', alpha=0.5)
        ax7.set_xlabel('α (公平性调节器)', fontsize=10)
        ax7.set_ylabel('NYK选秀收益', fontsize=10)
        ax7.set_title('(7) α参数与偏好向量的交互影响', fontsize=11, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        ax7.legend(fontsize=8, loc='best')
        
        # 8. 敏感性排名总结
        ax8 = fig.add_subplot(gs[2, 2])
        
        # 汇总所有敏感性指标
        all_sensitivities = []
        all_sensitivities.append(('α参数', alpha_results.get('sensitivity_index', 0.5)))
        for name, data in pv_results.items():
            all_sensitivities.append((f'PV-{name}', data['sensitivity']))
        for name, data in rookie_results.items():
            all_sensitivities.append((f'属性-{name}', abs(data['elasticity'])))
        all_sensitivities.append(('灵活性', flex_results['sensitivity']))
        
        # 排序并绘制
        all_sensitivities.sort(key=lambda x: x[1], reverse=True)
        top_10 = all_sensitivities[:10]
        
        names = [x[0] for x in top_10]
        values = [x[1] for x in top_10]
        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(names)))
        
        ax8.barh(names, values, color=colors)
        ax8.set_xlabel('敏感度/弹性', fontsize=10)
        ax8.set_title('(8) 参数敏感性Top 10排名', fontsize=11, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='x')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 综合敏感性分析图表已保存: {save_path}")
        
        return fig
    
    # =========================================================================
    # 报告生成
    # =========================================================================
    
    def generate_comprehensive_report(self) -> str:
        """生成全面的敏感性分析报告"""
        report = []
        report.append("\n" + "="*80)
        report.append("选秀策略公式敏感性分析 - 全面报告")
        report.append("="*80)
        
        # 公式1: α参数敏感性
        report.append("\n【公式1: 选秀权重公式 w_i^draft ∝ 1/(Q_i(t-1))^α】")
        report.append("-" * 80)
        
        alpha_data = self.sensitivity_results.get('alpha')
        if alpha_data:
            report.append(f"当前α值: {Config.ALPHA_DRAFT}")
            report.append(f"敏感度指数: {alpha_data.get('sensitivity_index', 0):.4f}")
            
            idx_current = np.argmin(np.abs(np.array(alpha_data['alpha_values']) - Config.ALPHA_DRAFT))
            current_gini = alpha_data['gini_coefficient'][idx_current]
            
            report.append(f"\n当前配置下:")
            report.append(f"  - 联盟基尼系数: {current_gini:.4f}")
            report.append(f"  - NYK选秀潜力: {alpha_data['nyk_draft_potential'][idx_current]:.4f}")
            report.append(f"  - 最弱5队平均权重: {alpha_data['bottom_5_avg_weight'][idx_current]:.4f}")
            report.append(f"  - 最强5队平均权重: {alpha_data['top_5_avg_weight'][idx_current]:.4f}")
            
            # 找到最优α
            idx_min_gini = np.argmin(alpha_data['gini_coefficient'])
            optimal_alpha = alpha_data['alpha_values'][idx_min_gini]
            optimal_gini = alpha_data['gini_coefficient'][idx_min_gini]
            
            report.append(f"\n最优α配置（公平性最大化）:")
            report.append(f"  - α = {optimal_alpha:.2f}")
            report.append(f"  - 基尼系数 = {optimal_gini:.4f}")
            report.append(f"  - 相比当前改善: {((current_gini - optimal_gini) / current_gini * 100):.1f}%")
        
        # 公式2: 偏好向量敏感性
        report.append("\n\n【公式2: 市场价值公式 V_mkt^i(A) = V_A^T · PV_i】")
        report.append("-" * 80)
        
        pv_data = self.sensitivity_results.get('preference_vector')
        if pv_data:
            report.append(f"当前NYK偏好向量: {self.nyk.pv}")
            report.append(f"\n各分量敏感性排名:")
            
            pv_sens = [(name, data['sensitivity']) for name, data in pv_data.items()]
            pv_sens.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (name, sens) in enumerate(pv_sens, 1):
                base_weight = pv_data[name]['base_weight']
                report.append(f"  {rank}. {name:8s}: 敏感度={sens:.4f}, 当前权重={base_weight:.3f}")
        
        # 公式3: 新秀属性敏感性
        report.append("\n\n【公式3: 球员价值向量 V_A = [AS, CS, SalEff, PW, Flex]^T】")
        report.append("-" * 80)
        
        rookie_data = self.sensitivity_results.get('rookie_attributes')
        if rookie_data:
            report.append("新秀各属性弹性系数:")
            
            elasticities = [(name, data['elasticity']) for name, data in rookie_data.items()]
            elasticities.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for rank, (name, elast) in enumerate(elasticities, 1):
                pv_weight = rookie_data[name]['pv_weight']
                report.append(f"  {rank}. {name:8s}: 弹性={elast:6.3f}, NYK权重={pv_weight:.3f}")
                
                # 解释弹性
                if elast > 1:
                    report.append(f"              → 高弹性：该属性提升1%，价值提升{elast:.1f}%")
                elif elast > 0.5:
                    report.append(f"              → 中等弹性")
                else:
                    report.append(f"              → 低弹性：投资回报率较低")
        
        # 公式4: 灵活性公式敏感性
        report.append("\n\n【公式4: 合同灵活性 Flex_A = e^(-rd(t))】")
        report.append("-" * 80)
        
        flex_data = self.sensitivity_results.get('flexibility')
        if flex_data:
            report.append(f"灵活性公式敏感度: {flex_data['sensitivity']:.4f}")
            report.append(f"\n关键合同年限节点:")
            
            # 找出关键节点
            rd_vals = flex_data['remaining_years']
            flex_vals = flex_data['flexibility_values']
            contrib_vals = flex_data['flexibility_contribution']
            
            key_points = [1, 2, 3, 4]
            for rd in key_points:
                idx = np.argmin(np.abs(np.array(rd_vals) - rd))
                report.append(f"  rd={rd}年: Flex={flex_vals[idx]:.4f}, "
                            f"对总价值贡献={contrib_vals[idx]:.4f}")
            
            report.append(f"\n结论: 剩余合同年限每增加1年，灵活性下降约{(1 - np.exp(-1))*100:.1f}%")
        
        # 公式5: 选秀收益公式
        report.append("\n\n【公式5: 选秀收益 ΔTQ_i^D(t) = Σ w_j^draft · n_{i,j}(t)】")
        report.append("-" * 80)
        
        draft_data = self.sensitivity_results.get('draft_position')
        if draft_data:
            report.append("前10顺位选秀权对NYK的价值:")
            
            for i in range(min(10, len(draft_data['pick_positions']))):
                pos = draft_data['pick_positions'][i]
                val = draft_data['value_to_nyk'][i]
                report.append(f"  第{pos:2d}顺位: 价值={val:.4f}")
            
            # 价值衰减分析
            val_1 = draft_data['value_to_nyk'][0]
            val_10 = draft_data['value_to_nyk'][9] if len(draft_data['value_to_nyk']) > 9 else 0
            val_30 = draft_data['value_to_nyk'][29] if len(draft_data['value_to_nyk']) > 29 else 0
            
            report.append(f"\n价值衰减:")
            report.append(f"  第1顺位 vs 第10顺位: 价值差{((val_1 - val_10) / val_1 * 100):.1f}%")
            if val_30 > 0:
                report.append(f"  第1顺位 vs 第30顺位: 价值差{((val_1 - val_30) / val_1 * 100):.1f}%")
        
        # 交互影响分析
        report.append("\n\n【公式交互影响分析】")
        report.append("-" * 80)
        
        cross_data = self.sensitivity_results.get('cross_alpha_pv')
        if cross_data:
            report.append("α参数与偏好向量的协同效应:")
            
            scenarios = cross_data['scenarios']
            alpha_vals = cross_data['alpha_values']
            
            for scenario_name, values in scenarios.items():
                idx_best = np.argmax(values)
                best_alpha = alpha_vals[idx_best]
                best_value = values[idx_best]
                
                report.append(f"\n  {scenario_name}:")
                report.append(f"    - 最优α = {best_alpha:.2f}")
                report.append(f"    - 最大收益 = {best_value:.4f}")
        
        # 总结与建议
        report.append("\n\n【总体敏感性排名与策略建议】")
        report.append("="*80)
        
        all_sensitivities = []
        
        if alpha_data:
            all_sensitivities.append(('α参数', alpha_data.get('sensitivity_index', 0)))
        
        if pv_data:
            for name, data in pv_data.items():
                all_sensitivities.append((f'PV_{name}', data['sensitivity']))
        
        if rookie_data:
            for name, data in rookie_data.items():
                all_sensitivities.append((f'属性_{name}', abs(data['elasticity'])))
        
        if flex_data:
            all_sensitivities.append(('Flex公式', flex_data['sensitivity']))
        
        all_sensitivities.sort(key=lambda x: x[1], reverse=True)
        
        report.append("\n最敏感参数Top 5:")
        for rank, (name, sens) in enumerate(all_sensitivities[:5], 1):
            report.append(f"  {rank}. {name:15s}: 敏感度/弹性 = {sens:.4f}")
        
        report.append("\n【针对NYK的策略建议】")
        report.append("基于敏感性分析结果:")
        
        # 根据PV配置给出建议
        if self.nyk.pv[0] > 0.35:  # AS权重高
            report.append("  1. NYK偏好即战力（AS权重高），应优先选择:")
            report.append("     - AS属性突出的新秀")
            report.append("     - 前10顺位选秀权（价值衰减显著）")
        
        if self.nyk.pv[1] > 0.15:  # CS权重较高
            report.append("  2. NYK重视商业价值（CS权重较高），可考虑:")
            report.append("     - 具有市场号召力的球员")
            report.append("     - 平衡竞技与商业的选秀策略")
        
        if self.nyk.pv[3] < 0.15:  # PW权重低
            report.append("  3. NYK对潜力重视度较低（PW权重低）:")
            report.append("     - 不建议囤积大量选秀权赌未来")
            report.append("     - 可用选秀权交易即战力球员")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序"""
    print(f"\n{'#'*80}")
    print(f"#{'选秀策略敏感性分析系统 - 增强版'.center(78)}#")
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
    print("\n[Step 3] 初始化增强版敏感性分析器...")
    analyzer = EnhancedDraftSensitivityAnalyzer(teams_dict)
    
    # Step 4: 生成综合图表
    print("\n[Step 4] 生成综合敏感性分析图表...")
    analyzer.plot_comprehensive_analysis("sensitivity_analysis_comprehensive.pdf")
    
    # Step 5: 生成全面报告
    print("\n[Step 5] 生成全面分析报告...")
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # 保存报告
    report_file = "sensitivity_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n✓ 分析报告已保存到: {report_file}")
    
    print("\n" + "="*80)
    print("全面敏感性分析完成！")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
