"""
改进版选秀策略敏感性分析
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

class ImprovedDraftSensitivityAnalyzer:
    """改进版选秀策略敏感性分析器"""
    
    def __init__(self, teams_dict: Dict[str, Team]):
        self.teams = list(teams_dict.values())
        self.nyk = teams_dict.get('NYK')
        self.sensitivity_metrics = {}
    
    def advanced_alpha_sensitivity(self, alpha_range: np.ndarray) -> Dict:
        """
        高级α敏感性分析 - 包含多种敏感性指标
        """
        results = {
            'alpha_values': [],
            'nyk_draft_weight': [],
            'nyk_draft_potential': [],
            'gini_coefficient': [],
            'sensitivity_derivatives': [],  # 导数敏感性
            'elasticity_scores': []         # 弹性系数
        }
        
        for alpha in alpha_range:
            calculate_draft_weights(self.teams, alpha=alpha)
            
            results['alpha_values'].append(alpha)
            results['nyk_draft_weight'].append(self.nyk.draft_weight)
            results['nyk_draft_potential'].append(self.nyk.draft_potential_score)
            
            # 计算基尼系数
            weights = [t.draft_weight for t in self.teams]
            gini = self._calculate_gini(weights)
            results['gini_coefficient'].append(gini)
        
        # 计算敏感性指标
        nyk_potentials = np.array(results['nyk_draft_potential'])
        alphas = np.array(results['alpha_values'])
        
        # 1. 导数敏感性 (一阶导数)
        derivatives = np.gradient(nyk_potentials, alphas)
        results['sensitivity_derivatives'] = derivatives.tolist()
        
        # 2. 弹性系数 (相对变化率)
        elasticities = []
        for i in range(len(alphas)-1):
            delta_alpha = (alphas[i+1] - alphas[i]) / alphas[i]
            delta_potential = (nyk_potentials[i+1] - nyk_potentials[i]) / nyk_potentials[i]
            elasticity = delta_potential / delta_alpha if delta_alpha != 0 else 0
            elasticities.append(elasticity)
        elasticities.append(elasticities[-1])  # 补充最后一个值
        results['elasticity_scores'] = elasticities
        
        return results
    
    def monte_carlo_sensitivity(self, n_simulations: int = 1000) -> Dict:
        """
        蒙特卡洛敏感性分析 - 处理不确定性
        """
        # 随机扰动参数
        alpha_samples = np.random.normal(Config.ALPHA_DRAFT, 0.3, n_simulations)
        alpha_samples = np.clip(alpha_samples, 0.5, 5.0)  # 约束范围
        
        nyk_outcomes = []
        
        for alpha in alpha_samples:
            # 临时修改并计算
            original_alpha = Config.ALPHA_DRAFT
            Config.ALPHA_DRAFT = alpha
            
            calculate_draft_weights(self.teams, alpha=alpha)
            nyk_outcomes.append(self.nyk.draft_potential_score)
            
            # 恢复原始值
            Config.ALPHA_DRAFT = original_alpha
        
        # 统计分析
        mean_outcome = np.mean(nyk_outcomes)
        std_outcome = np.std(nyk_outcomes)
        confidence_interval = stats.t.interval(0.95, len(nyk_outcomes)-1, 
                                             loc=mean_outcome, scale=stats.sem(nyk_outcomes))
        
        return {
            'mean_outcome': mean_outcome,
            'std_outcome': std_outcome,
            'confidence_interval': confidence_interval,
            'percentile_5th': np.percentile(nyk_outcomes, 5),
            'percentile_95th': np.percentile(nyk_outcomes, 95),
            'outcomes': nyk_outcomes,
            'alpha_samples': alpha_samples
        }
    
    def partial_correlation_analysis(self) -> Dict:
        """
        偏相关分析 - 控制其他变量的影响
        """
        # 构建特征矩阵
        features = []
        outcomes = []
        
        # 测试多个参数组合
        alpha_range = np.linspace(1.0, 3.0, 10)
        quality_range = np.linspace(0.2, 0.8, 10)
        
        for alpha in alpha_range:
            for quality in quality_range:
                # 模拟调整NYK质量
                original_quality = self.nyk.quality
                self.nyk.quality = quality
                
                calculate_draft_weights(self.teams, alpha=alpha)
                
                features.append([alpha, quality])
                outcomes.append(self.nyk.draft_potential_score)
                
                # 恢复
                self.nyk.quality = original_quality
        
        features = np.array(features)
        outcomes = np.array(outcomes)
        
        # 线性回归分析
        model = LinearRegression().fit(features, outcomes)
        
        return {
            'coefficients': model.coef_,
            'intercept': model.intercept_,
            'r_squared': model.score(features, outcomes),
            'feature_names': ['Alpha', 'NYK_Quality'],
            'predictions': model.predict(features)
        }
    
    def global_sensitivity_analysis(self) -> Dict:
        """
        全局敏感性分析 - Sobol指数
        """
        # 简化的Sobol分析
        param_ranges = {
            'alpha': (0.5, 5.0),
            'nyk_quality': (0.1, 0.9),
            'as_weight': (0.05, 0.5),
            'cs_weight': (0.05, 0.5)
        }
        
        # 生成样本点
        n_samples = 1000
        samples = {}
        for param, (low, high) in param_ranges.items():
            samples[param] = np.random.uniform(low, high, n_samples)
        
        outputs = []
        for i in range(n_samples):
            # 临时修改参数
            temp_alpha = samples['alpha'][i]
            temp_pv = self.nyk.pv.copy()
            temp_pv[0] = samples['as_weight'][i]  # AS权重
            temp_pv[1] = samples['cs_weight'][i]  # CS权重
            temp_pv = temp_pv / temp_pv.sum()    # 归一化
            
            original_pv = self.nyk.pv.copy()
            self.nyk.pv = temp_pv
            
            calculate_draft_weights(self.teams, alpha=temp_alpha)
            outputs.append(self.nyk.draft_potential_score)
            
            # 恢复
            self.nyk.pv = original_pv
        
        outputs = np.array(outputs)
        
        # 方差分解（简化版）
        total_variance = np.var(outputs)
        sensitivity_indices = {}
        
        for param in param_ranges.keys():
            # 用相关系数近似敏感性
            param_values = samples[param]
            correlation = np.corrcoef(param_values, outputs)[0, 1]
            sensitivity_indices[param] = abs(correlation)
        
        return {
            'sensitivity_indices': sensitivity_indices,
            'total_variance': total_variance,
            'param_ranges': param_ranges
        }
    
    def plot_advanced_sensitivity_analysis(self, save_path: str = "advanced_draft_sensitivity.png"):
        """绘制高级敏感性分析图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 1. α敏感性及导数
        alpha_range = np.linspace(0.5, 5.0, 100)
        alpha_results = self.advanced_alpha_sensitivity(alpha_range)
        
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot(alpha_results['alpha_values'], 
                        alpha_results['nyk_draft_potential'], 
                        'b-', linewidth=2, label='NYK Potential')
        line2 = ax1_twin.plot(alpha_results['alpha_values'], 
                         alpha_results['sensitivity_derivatives'], 
                         'r--', linewidth=2, label='Sensitivity Derivative')
        
        ax1.set_xlabel('Alpha (Fairness Parameter)')
        ax1.set_ylabel('NYK Draft Potential', color='b')
        ax1_twin.set_ylabel('Sensitivity Derivative', color='r')
        ax1.set_title('Alpha Sensitivity with Derivatives')
        ax1.grid(True, alpha=0.3)
        
        # 2. 蒙特卡洛分布
        mc_results = self.monte_carlo_sensitivity(n_simulations=1000)
        
        ax2 = axes[1]
        ax2.hist(mc_results['outcomes'], bins=50, density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        ax2.axvline(mc_results['mean_outcome'], color='red', linestyle='--', 
                   label=f'Mean: {mc_results["mean_outcome"]:.3f}')
        ax2.axvspan(mc_results['confidence_interval'][0], 
                   mc_results['confidence_interval'][1], 
                   alpha=0.3, color='yellow', label='95% CI')
        ax2.set_xlabel('NYK Draft Potential')
        ax2.set_ylabel('Density')
        ax2.set_title('Monte Carlo Distribution Analysis')
        ax2.legend()
        
        # 3. 偏相关分析
        pc_results = self.partial_correlation_analysis()
        
        ax3 = axes[2]
        scatter = ax3.scatter(pc_results['predictions'], 
                             [alpha_results['nyk_draft_potential'][i] 
                              for i in range(len(pc_results['predictions']))],
                             alpha=0.6)
        ax3.plot([pc_results['predictions'].min(), pc_results['predictions'].max()],
                [pc_results['predictions'].min(), pc_results['predictions'].max()], 
                'r--', linewidth=2)
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Actual Values')
        ax3.set_title(f'Partial Correlation (R² = {pc_results["r_squared"]:.3f})')
        
        # 4. 敏感性热力图
        ax4 = axes[3]
        sensitivity_data = np.column_stack([
            alpha_results['alpha_values'],
            alpha_results['nyk_draft_potential'],
            alpha_results['sensitivity_derivatives']
        ])
        im = ax4.imshow(sensitivity_data.T, aspect='auto', cmap='viridis')
        ax4.set_xlabel('Sample Index')
        ax4.set_ylabel('Parameter / Metric')
        ax4.set_yticks([0, 1, 2])
        ax4.set_yticklabels(['Alpha', 'Potential', 'Derivative'])
        ax4.set_title('Sensitivity Heatmap')
        plt.colorbar(im, ax=ax4)
        
        # 5. 全局敏感性分析
        gs_results = self.global_sensitivity_analysis()
        
        ax5 = axes[4]
        params = list(gs_results['sensitivity_indices'].keys())
        indices = list(gs_results['sensitivity_indices'].values())
        bars = ax5.bar(params, indices, color=['blue', 'red', 'green', 'orange'])
        ax5.set_ylabel('Sensitivity Index')
        ax5.set_title('Global Sensitivity Analysis (Sobol-like)')
        ax5.tick_params(axis='x', rotation=45)
        
        # 6. 属性敏感性对比
        attribute_ranges = {
            'AS': np.linspace(0.3, 0.9, 20),
            'CS': np.linspace(0.2, 0.8, 20),
            'PW': np.linspace(0.4, 1.0, 20),
        }
        attr_results = self.analyze_rookie_attributes(attribute_ranges)
        
        ax6 = axes[5]
        colors = ['blue', 'red', 'green']
        for idx, (attr_name, data) in enumerate(attr_results.items()):
            ax6.plot(data['attribute_values'], 
                    data['nyk_values'], 
                    linewidth=2, label=attr_name, color=colors[idx])
        ax6.set_xlabel('Attribute Value')
        ax6.set_ylabel('Value to NYK')
        ax6.set_title('Rookie Attributes Sensitivity')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 高级敏感性分析图表已保存到: {save_path}")
        
        return fig
    
    def generate_comprehensive_report(self) -> str:
        """生成综合敏感性分析报告"""
        report = []
        report.append("\n" + "="*100)
        report.append("综合选秀策略敏感性分析报告 - NYK视角")
        report.append("="*100)
        
        # 1. 局部敏感性分析
        alpha_range = np.linspace(0.5, 5.0, 100)
        alpha_results = self.advanced_alpha_sensitivity(alpha_range)
        
        report.append("\n【1. 局部敏感性分析 - Alpha参数】")
        current_alpha = Config.ALPHA_DRAFT
        current_idx = np.argmin(np.abs(np.array(alpha_results['alpha_values']) - current_alpha))
        current_potential = alpha_results['nyk_draft_potential'][current_idx]
        current_derivative = alpha_results['sensitivity_derivatives'][current_idx]
        
        report.append(f"  当前α = {current_alpha}, NYK潜力 = {current_potential:.4f}")
        report.append(f"  敏感性导数 = {current_derivative:.4f}")
        report.append(f"  说明: {'高度敏感' if abs(current_derivative) > 0.1 else '低度敏感'}")
        
        # 2. 全局敏感性分析
        gs_results = self.global_sensitivity_analysis()
        report.append("\n【2. 全局敏感性分析】")
        sorted_indices = sorted(gs_results['sensitivity_indices'].items(), 
                               key=lambda x: x[1], reverse=True)
        
        for i, (param, index) in enumerate(sorted_indices, 1):
            report.append(f"  {i}. {param}: {index:.3f}")
        
        # 3. 蒙特卡洛分析
        mc_results = self.monte_carlo_sensitivity()
        report.append("\n【3. 不确定性分析 (蒙特卡洛)】")
        report.append(f"  期望潜力: {mc_results['mean_outcome']:.4f}")
        report.append(f"  标准差: {mc_results['std_outcome']:.4f}")
        report.append(f"  95%置信区间: [{mc_results['confidence_interval'][0]:.4f}, {mc_results['confidence_interval'][1]:.4f}]")
        
        # 4. 风险评估
        report.append("\n【4. 风险评估】")
        risk_score = mc_results['std_outcome'] / mc_results['mean_outcome']
        if risk_score < 0.1:
            risk_level = "低风险"
        elif risk_score < 0.2:
            risk_level = "中风险"
        else:
            risk_level = "高风险"
        
        report.append(f"  变异系数: {risk_score:.3f} ({risk_level})")
        report.append(f"  5%分位数: {mc_results['percentile_5th']:.4f}")
        report.append(f"  95%分位数: {mc_results['percentile_95th']:.4f}")
        
        # 5. 优化建议
        report.append("\n【5. 参数优化建议】")
        derivatives = np.array(alpha_results['sensitivity_derivatives'])
        optimal_alpha_idx = np.argmin(np.abs(derivatives))  # 寻找最不敏感点
        optimal_alpha = alpha_results['alpha_values'][optimal_alpha_idx]
        
        report.append(f"  最优α建议: {optimal_alpha:.2f}")
        report.append(f"  稳定性建议: 当前α处于{'稳定' if abs(current_derivative) < 0.05 else '不稳定'}区域")
        
        report.append("\n" + "="*100)
        
        return "\n".join(report)

# 使用示例
def improved_main():
    """改进版主程序"""
    print("加载数据...")
    teams_dict = load_teams("fold_csv/team_quality.csv")
    
    print("初始化改进版分析器...")
    analyzer = ImprovedDraftSensitivityAnalyzer(teams_dict)
    
    print("生成高级分析图表...")
    analyzer.plot_advanced_sensitivity_analysis("improved_sensitivity_analysis.pdf")
    
    print("生成综合报告...")
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    with open("comprehensive_sensitivity_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

if __name__ == "__main__":
    improved_main()