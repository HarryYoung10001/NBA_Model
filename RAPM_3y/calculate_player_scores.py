#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NBA球员能力评分指标提取与计算程序
根据给定的公式提取和计算球员的6个关键指标
"""

import pandas as pd
import numpy as np

def calculate_player_metrics(input_file, output_file):
    """
    从CSV文件中提取球员数据并计算所需指标
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
    """
    # 读取CSV文件
    print(f"正在读取数据文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 提取Offense值（去除括号中的百分位数）
    # 例如 "4.6 (99)" -> 4.6
    df['RAPM_offense'] = df['Offense'].apply(lambda x: float(str(x).split('(')[0].strip()) if pd.notna(x) and str(x) != '' else np.nan)
    
    # 提取Defense值（去除括号中的百分位数）
    # 例如 "-2.1 (98)" -> -2.1
    df['RAPM_defense'] = df['Defense'].apply(lambda x: float(str(x).split('(')[0].strip()) if pd.notna(x) and str(x) != '' else np.nan)
    
    # 直接提取USG_PCT
    df['USG_percent'] = df['USG_PCT']
    
    # 直接提取PLUS_MINUS
    df['PLUS_MINUS_value'] = df['PLUS_MINUS']
    
    # 直接提取ATTENDANCE_RATE
    df['Attendance'] = df['ATTENDANCE_RATE']
    
    # 计算Performance = (REB + AST + STL + BLK + PTS - TOV) / GP
    # 注意处理缺失值的情况
    def calculate_performance(row):
        try:
            if pd.notna(row['GP']) and row['GP'] > 0:
                numerator = (
                    row['REB'] + row['AST'] + row['STL'] + 
                    row['BLK'] + row['PTS'] - row['TOV']
                )
                return numerator / row['GP']
            else:
                return np.nan
        except:
            return np.nan
    
    df['Performance'] = df.apply(calculate_performance, axis=1)
    
    # 选择需要的列
    output_columns = [
        'Player',                # 球员姓名
        'Team',                  # 球队
        'RAPM_offense',          # RAPM^{offense}
        'RAPM_defense',          # RAPM^{defense}
        'USG_percent',           # USG%
        'PLUS_MINUS_value',      # PLUS_MINUS
        'Attendance',            # Attendance（出勤率）
        'Performance'            # Performance（综合表现）
    ]
    
    # 创建输出数据框
    output_df = df[output_columns].copy()
    
    # 保存到新的CSV文件
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n数据处理完成！")
    print(f"输出文件已保存至: {output_file}")
    
    # 显示统计信息
    print(f"\n总共处理了 {len(output_df)} 名球员的数据")
    print(f"有效数据记录: {output_df.dropna(subset=['Performance']).shape[0]} 条")
    
    # 显示前几行数据作为示例
    print("\n前5名球员的数据预览：")
    print(output_df.head().to_string())
    
    return output_df

if __name__ == "__main__":
    # 定义输入输出文件路径
    input_file = "nba_combined_data.csv"
    output_file = "player_metrics.csv"
    
    # 执行数据处理
    result_df = calculate_player_metrics(input_file, output_file)
    
    print("\n" + "="*60)
    print("指标说明：")
    print("="*60)
    print("RAPM_offense:    进攻真实正负值（Offense）")
    print("RAPM_defense:    防守真实正负值（Defense）")
    print("USG_percent:     使用率百分比（USG%）")
    print("PLUS_MINUS_value: 正负值（PLUS_MINUS）")
    print("Attendance:      出勤率（ATTENDANCE_RATE）")
    print("Performance:     综合表现 = (REB+AST+STL+BLK+PTS-TOV)/GP")
    print("="*60)

