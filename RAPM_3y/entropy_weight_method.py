#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用熵权法计算NBA球员能力评分指标的权重
步骤：
1. 数据归一化：
   - 正向指标（越大越好）：(x - min) / (max - min)
   - 逆向指标（越小越好）：(max - x) / (max - min)
2. 使用熵权法计算各指标权重
"""

import pandas as pd
import numpy as np

def normalize_data(df, positive_columns, negative_columns):
    """
    对指定列进行归一化处理
    
    参数:
        df: 数据框
        positive_columns: 正向指标列名列表（越大越好）
        negative_columns: 逆向指标列名列表（越小越好）
    
    返回:
        归一化后的数据框
    """
    df_normalized = df.copy()
    
    print("="*60)
    print("数据归一化处理")
    print("="*60)
    
    # 处理正向指标
    print("\n【正向指标归一化】(越大越好)")
    print("公式: (x - min) / (max - min)")
    print("-"*60)
    
    for col in positive_columns:
        # 删除缺失值以计算最小值和最大值
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            print(f"警告: {col} 列没有有效数据，跳过归一化")
            continue
            
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        print(f"\n{col}:")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")
        
        # 归一化
        if max_val - min_val != 0:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
            print(f"  归一化完成")
        else:
            print(f"  警告: 最大值等于最小值，该列数据无变化")
            df_normalized[col] = 0
    
    # 处理逆向指标
    print("\n" + "="*60)
    print("【逆向指标归一化】(越小越好)")
    print("公式: (max - x) / (max - min)")
    print("-"*60)
    
    for col in negative_columns:
        # 删除缺失值以计算最小值和最大值
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            print(f"警告: {col} 列没有有效数据，跳过归一化")
            continue
            
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        print(f"\n{col}:")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")
        
        # 逆向归一化
        if max_val - min_val != 0:
            df_normalized[col] = (max_val - df[col]) / (max_val - min_val)
            print(f"  逆向归一化完成")
        else:
            print(f"  警告: 最大值等于最小值，该列数据无变化")
            df_normalized[col] = 0
    
    return df_normalized

def calculate_entropy_weights(df, columns):
    """
    使用熵权法计算各指标的权重
    
    参数:
        df: 归一化后的数据框
        columns: 需要计算权重的列名列表
    
    返回:
        包含权重、熵值和差异系数的元组
    """
    # 只使用完整数据的行
    df_clean = df[columns].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("没有完整的数据行可用于计算熵权")
    
    m, n = df_clean.shape  # m个样本，n个指标
    print(f"\n用于计算熵权的样本数: {m}")
    print(f"指标数: {n}")
    
    # 步骤1: 计算各指标下每个样本的比重 p_ij
    # 为避免对数计算时出现0，需要对归一化后的数据进行微小调整
    epsilon = 1e-10
    df_adjusted = df_clean + epsilon
    
    # 计算每列的总和
    col_sums = df_adjusted.sum(axis=0)
    
    # 计算比重矩阵
    p_matrix = df_adjusted / col_sums
    
    print("\n"+"="*60)
    print("熵权法计算过程")
    print("="*60)
    
    # 步骤2: 计算各指标的熵值
    k = 1 / np.log(m)  # 熵值计算常数
    entropies = {}
    
    print(f"\n熵值常数 k = 1/ln(m) = 1/ln({m}) = {k:.6f}")
    print("\n熵值计算公式: e_j = -k × Σ(p_ij × ln(p_ij))")
    print("-"*60)
    
    for col in columns:
        # 计算熵值 e_j = -k * sum(p_ij * ln(p_ij))
        p_col = p_matrix[col]
        entropy_value = -k * np.sum(p_col * np.log(p_col))
        entropies[col] = entropy_value
        print(f"{col}:")
        print(f"  熵值 (e_j): {entropy_value:.6f}")
    
    # 步骤3: 计算差异系数（信息效用值）
    differences = {}
    print("\n" + "="*60)
    print("差异系数计算")
    print("="*60)
    print("公式: d_j = 1 - e_j")
    print("-"*60)
    
    for col in columns:
        # d_j = 1 - e_j
        diff_value = 1 - entropies[col]
        differences[col] = diff_value
        print(f"{col}:")
        print(f"  d_j = 1 - {entropies[col]:.6f} = {diff_value:.6f}")
    
    # 步骤4: 计算权重
    sum_diff = sum(differences.values())
    weights = {}
    
    print("\n" + "="*60)
    print("权重计算")
    print("="*60)
    print(f"公式: w_j = d_j / Σd_j")
    print(f"差异系数总和: Σd_j = {sum_diff:.6f}")
    print("-"*60)
    
    for col in columns:
        # w_j = d_j / sum(d_j)
        weight_value = differences[col] / sum_diff
        weights[col] = weight_value
        print(f"{col}:")
        print(f"  w_j = {differences[col]:.6f} / {sum_diff:.6f} = {weight_value:.6f} ({weight_value*100:.2f}%)")
    
    # 验证权重和为1
    weight_sum = sum(weights.values())
    print(f"\n权重总和验证: {weight_sum:.6f}")
    
    if abs(weight_sum - 1.0) > 1e-6:
        print("警告: 权重总和不等于1！")
    else:
        print("✓ 权重总和正确")
    
    return weights, entropies, differences

def main():
    # 读取数据
    input_file = "csv_fold/player_metrics.csv"
    print(f"正在读取数据文件: {input_file}\n")
    df = pd.read_csv(input_file)
    
    # 定义正向指标（越大越好）
    positive_metrics = [
        'RAPM_offense',      # 进攻贡献越高越好
        'USG_percent',       # 使用率越高说明球员越重要
        'PLUS_MINUS_value',  # 正负值越大越好
        'Attendance',        # 出勤率越高越好
        'Performance'        # 表现越好越好
    ]
    
    # 定义逆向指标（越小越好）
    negative_metrics = [
        'RAPM_defense'       # 防守失分越少越好
    ]
    
    # 所有指标
    all_metrics = positive_metrics + negative_metrics
    
    # 显示指标分类
    print("="*60)
    print("指标分类说明")
    print("="*60)
    print(f"\n正向指标（越大越好）: {len(positive_metrics)}个")
    for i, metric in enumerate(positive_metrics, 1):
        print(f"  {i}. {metric}")
    
    print(f"\n逆向指标（越小越好）: {len(negative_metrics)}个")
    for i, metric in enumerate(negative_metrics, 1):
        print(f"  {i}. {metric}")
    
    # 显示原始数据统计
    print("\n" + "="*60)
    print("原始数据统计")
    print("="*60)
    print(df[all_metrics].describe())
    
    # 步骤1: 数据归一化
    df_normalized = normalize_data(df, positive_metrics, negative_metrics)
    
    # 保存归一化后的数据
    normalized_output = "csv_fold/player_metrics_normalized.csv"
    
    # 保留球员姓名和队伍信息
    output_df = df[['Player', 'Team']].copy()
    for col in all_metrics:
        output_df[col + '_normalized'] = df_normalized[col]
    
    output_df.to_csv(normalized_output, index=False, encoding='utf-8-sig')
    print(f"\n归一化数据已保存至: {normalized_output}")
    
    # 步骤2: 使用熵权法计算权重
    weights, entropies, differences = calculate_entropy_weights(
        df_normalized, 
        all_metrics
    )
    
    # 创建权重结果数据框
    weights_df = pd.DataFrame({
        '指标': [
            'ω₀ (RAPM_offense)',
            'ω₁ (RAPM_defense)',
            'ω₂ (USG_percent)',
            'ω₃ (PLUS_MINUS_value)',
            'ω₄ (Attendance)',
            'ω₅ (Performance)'
        ],
        '权重符号': ['ω₀', 'ω₁', 'ω₂', 'ω₃', 'ω₄', 'ω₅'],
        '对应指标': all_metrics,
        '指标类型': ['正向', '逆向', '正向', '正向', '正向', '正向'],
        '熵值 (e_j)': [entropies[col] for col in all_metrics],
        '差异系数 (d_j)': [differences[col] for col in all_metrics],
        '权重 (w_j)': [weights[col] for col in all_metrics],
        '权重百分比': [f"{weights[col]*100:.2f}%" for col in all_metrics]
    })
    
    # 保存权重结果
    weights_output = "csv_fold/entropy_weights.csv"
    weights_df.to_csv(weights_output, index=False, encoding='utf-8-sig')
    print(f"\n权重计算结果已保存至: {weights_output}")
    
    # 显示最终权重表
    print("\n" + "="*60)
    print("最终权重结果")
    print("="*60)
    print(weights_df.to_string(index=False))
    
    # 输出公式
    print("\n" + "="*60)
    print("球员能力评分公式")
    print("="*60)
    print("\n注意: 所有指标均已归一化至 [0,1] 区间")
    print("      RAPM_defense 已进行逆向归一化处理（越小越好 → 越大越好）")
    print("\n综合评分公式:")
    print("\nA_i = ", end="")
    for i, col in enumerate(all_metrics):
        w = weights[col]
        if i > 0:
            print(f" + ", end="")
        # 标注归一化后的指标
        print(f"{w:.6f} × {col}_normalized_i", end="")
    print("\n")
    
    print("其中:")
    for col in all_metrics:
        indicator_type = "正向" if col in positive_metrics else "逆向"
        if col in positive_metrics:
            formula = "(x - min) / (max - min)"
        else:
            formula = "(max - x) / (max - min)"
        print(f"  {col}_normalized = {formula}  [{indicator_type}指标]")
    
    return df_normalized, weights_df

if __name__ == "__main__":
    df_normalized, weights_df = main()
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)

