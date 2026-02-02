#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用熵权法计算得到的权重为每个球员计算综合能力评分（包含排名）
支持正向指标和逆向指标的不同归一化方式
添加了AGE列
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
        归一化后的数据框和归一化参数
    """
    df_normalized = df.copy()
    normalization_params = {}
    
    # 处理正向指标
    print("\n【正向指标归一化】(越大越好)")
    print("公式: (x - min) / (max - min)")
    print("-" * 60)
    
    for col in positive_columns:
        # 删除缺失值以计算最小值和最大值
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            print(f"警告: {col} 列没有有效数据，跳过归一化")
            continue
            
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        normalization_params[col] = {
            'min': min_val, 
            'max': max_val, 
            'type': 'positive'
        }
        
        print(f"{col}: min={min_val:.4f}, max={max_val:.4f}")
        
        # 归一化
        if max_val - min_val != 0:
            df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    
    # 处理逆向指标
    print("\n【逆向指标归一化】(越小越好)")
    print("公式: (max - x) / (max - min)")
    print("-" * 60)
    
    for col in negative_columns:
        # 删除缺失值以计算最小值和最大值
        valid_data = df[col].dropna()
        
        if len(valid_data) == 0:
            print(f"警告: {col} 列没有有效数据，跳过归一化")
            continue
            
        min_val = valid_data.min()
        max_val = valid_data.max()
        
        normalization_params[col] = {
            'min': min_val, 
            'max': max_val, 
            'type': 'negative'
        }
        
        print(f"{col}: min={min_val:.4f}, max={max_val:.4f}")
        
        # 逆向归一化
        if max_val - min_val != 0:
            df_normalized[col] = (max_val - df[col]) / (max_val - min_val)
        else:
            df_normalized[col] = 0
    
    return df_normalized, normalization_params

def calculate_player_scores(weights_file, metrics_file, nba_data_file, output_file):
    """
    计算每个球员的综合能力评分并添加排名
    
    参数:
        weights_file: 权重文件路径
        metrics_file: 球员指标文件路径
        nba_data_file: NBA综合数据文件路径（包含AGE列）
        output_file: 输出文件路径
    """
    print("="*60)
    print("NBA球员综合能力评分计算（含排名和年龄）")
    print("="*60)
    
    # 1. 读取权重数据
    print("\n步骤1: 读取权重数据")
    weights_df = pd.read_csv(weights_file)
    print(weights_df)
    
    # 提取权重值
    weights = {}
    for _, row in weights_df.iterrows():
        metric_name = row['对应指标']
        weight_value = row['权重 (w_j)']
        weights[metric_name] = weight_value
    
    print("\n提取的权重值:")
    for metric, weight in weights.items():
        print(f"  {metric}: {weight:.6f}")
    
    # 2. 读取球员指标数据
    print("\n步骤2: 读取球员指标数据")
    players_df = pd.read_csv(metrics_file)
    print(f"总共 {len(players_df)} 名球员")
    
    # 3. 读取NBA综合数据以获取AGE列
    print("\n步骤3: 读取NBA综合数据（获取AGE列）")
    nba_data_df = pd.read_csv(nba_data_file)
    # 只保留Player和AGE列
    age_df = nba_data_df[['Player', 'AGE']].copy()
    print(f"从NBA数据中读取了 {len(age_df)} 名球员的年龄信息")
    
    # 合并年龄数据到球员数据
    players_df = players_df.merge(age_df, on='Player', how='left')
    print(f"成功合并年龄数据")
    
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
    print("\n指标分类:")
    print(f"  正向指标: {', '.join(positive_metrics)}")
    print(f"  逆向指标: {', '.join(negative_metrics)}")
    
    # 4. 数据归一化
    print("\n步骤4: 数据归一化")
    players_normalized, norm_params = normalize_data(
        players_df, 
        positive_metrics, 
        negative_metrics
    )
    
    print("\n归一化参数汇总:")
    for col in all_metrics:
        if col in norm_params:
            params = norm_params[col]
            indicator_type = "正向" if params['type'] == 'positive' else "逆向"
            print(f"  {col} [{indicator_type}]: min={params['min']:.4f}, max={params['max']:.4f}")
    
    # 5. 计算综合得分
    print("\n步骤5: 计算综合得分")
    print("公式: A_i = Σ(w_j × 归一化指标_j)")
    print("注: RAPM_defense 已进行逆向归一化处理")
    
    # 初始化得分列
    players_df['Athletic_Score'] = 0.0
    
    # 对每个指标应用权重并累加
    for metric in all_metrics:
        if metric in weights:
            # 使用归一化后的值乘以权重
            players_df['Athletic_Score'] += (
                players_normalized[metric].fillna(0) * weights[metric]
            )
    
    # 处理缺失值：如果某个球员的所有指标都缺失，则总分为NaN
    all_missing = players_df[all_metrics].isna().all(axis=1)
    players_df.loc[all_missing, 'Athletic_Score'] = np.nan
    
    # 6. 添加排名列
    print("\n步骤6: 添加排名")
    # 使用rank方法，降序排列（分数高的排名靠前），缺失值排在最后
    players_df['Rank'] = players_df['Athletic_Score'].rank(
        ascending=False, 
        method='min', 
        na_option='bottom'
    ).astype(int)
    
    # 7. 创建输出数据框（添加AGE列）
    output_df = pd.DataFrame({
        'Rank': players_df['Rank'],
        'Player': players_df['Player'],
        'Team': players_df['Team'],
        'AGE': players_df['AGE'],  # 添加AGE列
        'RAPM_offense': players_df['RAPM_offense'],
        'RAPM_defense': players_df['RAPM_defense'],
        'USG_percent': players_df['USG_percent'],
        'PLUS_MINUS_value': players_df['PLUS_MINUS_value'],
        'Attendance': players_df['Attendance'],
        'Performance': players_df['Performance'],
        'Athletic_Score': players_df['Athletic_Score']
    })
    
    # 按排名排序
    output_df_sorted = output_df.sort_values('Rank')
    
    # 8. 保存结果
    output_df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n球员得分（含排名和年龄）已保存至: {output_file}")
    
    # 9. 显示统计信息
    print("\n" + "="*60)
    print("得分统计信息")
    print("="*60)
    valid_scores = output_df_sorted['Athletic_Score'].dropna()
    print(f"有效得分数量: {len(valid_scores)}")
    print(f"平均分: {valid_scores.mean():.6f}")
    print(f"最高分: {valid_scores.max():.6f}")
    print(f"最低分: {valid_scores.min():.6f}")
    print(f"标准差: {valid_scores.std():.6f}")
    print(f"中位数: {valid_scores.median():.6f}")
    
    # 10. 显示Top 20球员
    print("\n" + "="*60)
    print("Top 20 球员排名")
    print("="*60)
    top_20 = output_df_sorted.head(20)
    display_df = top_20[['Rank', 'Player', 'Team', 'AGE', 'Athletic_Score']].copy()
    print(display_df.to_string(index=False))
    
    # 11. 显示详细得分分解（前5名）
    print("\n" + "="*60)
    print("前5名球员得分详细分解")
    print("="*60)
    
    for idx, (i, row) in enumerate(output_df_sorted.head(5).iterrows(), 1):
        age_info = f", 年龄: {row['AGE']:.0f}" if pd.notna(row['AGE']) else ""
        print(f"\n排名 {row['Rank']}: {row['Player']} ({row['Team']}{age_info}) - 总分: {row['Athletic_Score']:.6f}")
        print("-" * 60)
        
        # 计算各指标的贡献
        total_contribution = 0
        
        for metric in all_metrics:
            # 检查指标值是否有效
            if pd.notna(row[metric]):
                # 检查归一化参数是否存在
                if metric not in norm_params:
                    print(f"  警告: {metric} 的归一化参数不存在")
                    continue
                
                # 获取归一化参数
                params = norm_params[metric]
                min_value = params['min']
                max_value = params['max']
                indicator_type = params['type']
                
                # 根据指标类型计算归一化值
                if max_value != min_value:
                    if indicator_type == 'positive':
                        # 正向指标: (x - min) / (max - min)
                        normalized = (row[metric] - min_value) / (max_value - min_value)
                    else:
                        # 逆向指标: (max - x) / (max - min)
                        normalized = (max_value - row[metric]) / (max_value - min_value)
                else:
                    normalized = 0
                
                # 计算贡献值
                if metric in weights:
                    contribution = normalized * weights[metric]
                    total_contribution += contribution
                    
                    type_label = "正向" if indicator_type == 'positive' else "逆向"
                    print(f"  {metric:20s} [{type_label}]: 原始值 {row[metric]:8.4f} → 归一化 {normalized:6.4f} → 贡献 {contribution:.6f} (权重 {weights[metric]:.4f})")
                else:
                    print(f"  警告: {metric} 没有对应的权重")
        
        print(f"  {'='*20}")
        print(f"  {'总计':20s}: {'':8s}    {'':6s}    {total_contribution:.6f}")
    
    # 12. 显示排名分布
    print("\n" + "="*60)
    print("排名分布统计")
    print("="*60)
    
    # 按得分区间统计
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
    output_df_sorted['Score_Range'] = pd.cut(
        output_df_sorted['Athletic_Score'], 
        bins=bins, 
        labels=labels, 
        include_lowest=True
    )
    
    score_distribution = output_df_sorted['Score_Range'].value_counts().sort_index()
    print("\n得分区间分布:")
    for range_label, count in score_distribution.items():
        percentage = (count / len(valid_scores)) * 100
        print(f"  {range_label}: {count:3d} 名球员 ({percentage:5.2f}%)")
    
    return output_df_sorted

def main():
    """
    主函数
    """
    # 文件路径配置
    weights_file = 'entropy_weights.csv'
    metrics_file = 'player_metrics.csv'
    nba_data_file = 'nba_combined_data.csv'  # 新增：包含AGE列的文件
    output_file = 'player_scores.csv'
    
    # 计算球员得分
    result_df = calculate_player_scores(weights_file, metrics_file, nba_data_file, output_file)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    print(f"\n输出文件: {output_file}")
    print(f"总球员数: {len(result_df)}")
    print(f"包含列: Rank, Player, Team, AGE, RAPM_offense, RAPM_defense,")
    print(f"        USG_percent, PLUS_MINUS_value, Attendance,")
    print(f"        Performance, Athletic_Score")
    print(f"\n重要说明:")
    print(f"  - 新增了AGE列（从nba_combined_data.csv中获取）")
    print(f"  - RAPM_defense 已使用逆向归一化: (max - x) / (max - min)")
    print(f"  - 其他指标使用正向归一化: (x - min) / (max - min)")
    print(f"  - 所有归一化值均在 [0, 1] 区间内")
    
    return result_df

if __name__ == "__main__":
    result = main()

