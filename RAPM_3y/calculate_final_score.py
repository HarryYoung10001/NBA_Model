#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算球员最终得分
使用公式: TS_i = α * AS_i * Potential_weight + (1-α) * CS_i
其中:
- AS_i = Athletic_Score (运动能力得分)
- CS_i = final_commercial_score (商业得分)
- Potential_weight = potential (潜力权重)
- α: 运动能力和商业价值的权重参数
- k: 预留参数（可用于未来扩展）
"""

import pandas as pd
import sys

def calculate_final_scores(input_file, output_file, alpha=0.7, k=1.0):
    """
    计算球员的最终得分
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        alpha: 运动能力权重参数 (默认0.7)
        k: 预留参数 (默认1.0)
    """
    # 读取数据
    print(f"正在读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 显示数据基本信息
    print(f"共读取 {len(df)} 名球员的数据")
    print(f"\n参数设置:")
    print(f"  α (alpha) = {alpha}")
    print(f"  k = {k}")
    
    # 计算最终得分
    # TS_i = α * AS_i * Potential_weight + (1-α) * CS_i
    df['final_score'] = (
        alpha * df['Athletic_Score'] * df['potential'] + 
        (1 - alpha) * df['final_commercial_score']
    )
    
    # 选择输出列
    output_df = df[['Player', 'Team', 'final_score']].copy()
    
    # 按最终得分降序排序
    output_df = output_df.sort_values('final_score', ascending=False)
    
    # 保存结果
    output_df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n结果已保存到: {output_file}")
    
    # 显示前10名球员
    print("\n最终得分排名前10的球员:")
    print(output_df.head(10).to_string(index=False))
    
    # 显示统计信息
    print(f"\n最终得分统计:")
    print(f"  最高分: {output_df['final_score'].max():.6f}")
    print(f"  最低分: {output_df['final_score'].min():.6f}")
    print(f"  平均分: {output_df['final_score'].mean():.6f}")
    print(f"  中位数: {output_df['final_score'].median():.6f}")
    
    return output_df

if __name__ == "__main__":
    # 默认参数
    alpha = 0.5  # 可以通过命令行参数修改
    k = 3.0      # 预留参数
    
    # 如果提供了命令行参数，则使用命令行参数
    if len(sys.argv) > 1:
        alpha = float(sys.argv[1])
    if len(sys.argv) > 2:
        k = float(sys.argv[2])
    
    # 文件路径
    input_file = "csv_fold/3scores.csv"
    output_file = "csv_fold/final_score.csv"
    
    # 计算并保存结果
    calculate_final_scores(input_file, output_file, alpha, k)
    
    print("\n提示: 可以通过命令行参数修改α和k的值")
    print(f"用alpha] [k]")
    print(f"例如: python {sys.argv[0]} 0.5 3.0")

