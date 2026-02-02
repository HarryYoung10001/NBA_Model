"""
计算Team Quality (TQ)
从match_stat.csv读取数据，计算TQ，从nba_team_stat.csv获取胜率，保存到TQ_stat.csv
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_team_quality(df_team: pd.DataFrame, p: float = 2.0, q: float = 2.0) -> float:
    """
    计算单个球队的Team Quality
    
    TQ = (Σ AR_i · (RAPM^off_i)^p)^(1/p) - (Σ AR_i · (RAPM^def_i)^q)^(1/q)
    
    参数:
        df_team: 单个球队的球员数据
        p: 进攻RAPM的指数参数
        q: 防守RAPM的指数参数
    
    返回:
        Team Quality值
    """
    # 过滤掉出勤率为0的球员
    df_active = df_team[df_team['attendance_rate'] > 0].copy()
    
    if len(df_active) == 0:
        return np.nan
    
    # 计算进攻部分
    # 只考虑正的进攻RAPM值
    df_active['off_positive'] = df_active['rapm_off'].apply(lambda x: max(x, 0))
    
    offense_sum = (df_active['attendance_rate'] * (df_active['off_positive'] ** p)).sum()
    if offense_sum > 0:
        offense_component = offense_sum ** (1/p)
    else:
        offense_component = 0
    
    # 计算防守部分
    # 防守RAPM：负数表示好的防守，所以我们取绝对值
    df_active['def_absolute'] = df_active['rapm_def'].apply(lambda x: abs(min(x, 0)))
    
    defense_sum = (df_active['attendance_rate'] * (df_active['def_absolute'] ** q)).sum()
    if defense_sum > 0:
        defense_component = defense_sum ** (1/q)
    else:
        defense_component = 0
    
    # 计算TQ
    tq = offense_component - defense_component
    
    return tq

def load_team_win_percentages(file_path: str = None) -> pd.DataFrame:
    """
    从CSV文件加载球队胜率数据
    
    参数:
        file_path: nba_team_stat.csv文件路径
    
    返回:
        包含球队胜率的DataFrame
    """
    # 尝试多个可能的路径
    if file_path is None:
        possible_paths = [
            'TQ_stat/nba_team_stat.csv',
            '/mnt/user-data/outputs/nba_team_stat.csv',
            'nba_team_stat.csv',
            'output/nba_team_stat.csv',
        ]
    else:
        possible_paths = [file_path]
    
    import os
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logging.info(f"从 {path} 加载了 {len(df)} 条球队战绩记录")
                
                # 确保year是字符串类型以便匹配
                df['year'] = df['year'].astype(str)
                
                return df[['team', 'year', 'win_pct']]
                
            except Exception as e:
                logging.error(f"加载 {path} 时出错: {str(e)}")
                continue
    
    logging.error(f"无法找到球队战绩文件，尝试的路径: {possible_paths}")
    return pd.DataFrame()

def calculate_all_team_qualities(df: pd.DataFrame, p: float = 2.0, q: float = 2.0) -> pd.DataFrame:
    """
    计算所有球队的Team Quality
    
    参数:
        df: match_stat.csv的数据
        p: 进攻RAPM的指数参数
        q: 防守RAPM的指数参数
    
    返回:
        包含(team, year, TQ)的DataFrame
    """
    results = []
    
    # 按年份和球队分组
    grouped = df.groupby(['year', 'team'])
    
    total_teams = len(grouped)
    logging.info(f"开始计算 {total_teams} 个球队-赛季组合的TQ...")
    
    for idx, ((year, team), group) in enumerate(grouped, 1):
        tq = calculate_team_quality(group, p=p, q=q)
        
        results.append({
            'team': team,
            'year': year,
            'TQ': tq
        })
        
        if idx % 10 == 0:
            logging.info(f"已计算 {idx}/{total_teams} 个球队...")
    
    df_result = pd.DataFrame(results)
    logging.info(f"TQ计算完成! 共 {len(df_result)} 条记录")
    
    return df_result

def add_win_percentages(df_tq: pd.DataFrame, df_team_stats: pd.DataFrame) -> pd.DataFrame:
    """
    为TQ数据添加胜率信息
    
    参数:
        df_tq: 包含TQ的DataFrame
        df_team_stats: 包含球队胜率的DataFrame
    
    返回:
        合并后的DataFrame
    """
    logging.info("正在合并TQ和胜率数据...")
    
    # 确保year列类型一致
    df_tq['year'] = df_tq['year'].astype(str)
    df_team_stats['year'] = df_team_stats['year'].astype(str)
    
    # 合并数据
    df_merged = df_tq.merge(
        df_team_stats,
        on=['team', 'year'],
        how='left'
    )
    
    # 统计匹配情况
    matched = df_merged['win_pct'].notna().sum()
    total = len(df_merged)
    logging.info(f"成功匹配 {matched}/{total} 条记录 ({matched/total*100:.1f}%)")
    
    if matched < total:
        unmatched = df_merged[df_merged['win_pct'].isna()][['team', 'year']]
        logging.warning(f"\n未匹配的记录:\n{unmatched}")
    
    return df_merged

def main(p: float = 2.0, q: float = 2.0):
    """
    主函数
    
    参数:
        p: 进攻RAPM的指数参数（默认2.0）
        q: 防守RAPM的指数参数（默认2.0）
    """
    # 1. 加载match_stat.csv
    logging.info("步骤1: 加载匹配数据...")
    
    import os
    possible_input_paths = [
        'TQ_stat/match_stat.csv',
        '/mnt/user-data/outputs/match_stat.csv',
        'match_stat.csv',
        'output/match_stat.csv',
    ]
    
    df = None
    for path in possible_input_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logging.info(f"从 {path} 加载了 {len(df)} 条记录")
                break
            except Exception as e:
                logging.error(f"加载 {path} 时出错: {str(e)}")
                continue
    
    if df is None:
        logging.error(f"无法加载match_stat.csv，尝试的路径: {possible_input_paths}")
        logging.info("请先运行 match_player.py 生成该文件")
        return
    
    # 2. 计算Team Quality
    logging.info(f"\n步骤2: 计算Team Quality (p={p}, q={q})...")
    df_tq = calculate_all_team_qualities(df, p=p, q=q)
    
    # 3. 加载球队胜率数据
    logging.info("\n步骤3: 加载球队胜率数据...")
    df_team_stats = load_team_win_percentages()
    
    if df_team_stats.empty:
        logging.error("无法加载球队胜率数据")
        logging.info("请先运行 nba_get.py 生成 nba_team_stat.csv")
        return
    
    # 4. 合并TQ和胜率
    logging.info("\n步骤4: 合并TQ和胜率...")
    df_final = add_win_percentages(df_tq, df_team_stats)
    
    # 5. 按TQ排序
    df_final = df_final.sort_values(['year', 'TQ'], ascending=[True, False]).reset_index(drop=True)
    
    # 6. 保存结果
    # 尝试多个可能的输出路径
    possible_output_dirs = [
        'TQ_stat',
        '/mnt/user-data/outputs',
        'output',
        '.'  # 当前目录
    ]
    
    output_dir = None
    for dir_path in possible_output_dirs:
        if os.path.exists(dir_path) or dir_path == '.':
            output_dir = dir_path
            break
    
    if output_dir is None:
        # 创建TQ_stat目录
        output_dir = 'TQ_stat'
        os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'TQ_stat.csv')
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"\n数据已保存到 {output_file}")
    
    # 7. 显示结果
    print("\n" + "="*80)
    print("Team Quality 统计结果")
    print("="*80)
    
    print(f"\n参数设置: p={p}, q={q}")
    
    print("\n前20名球队:")
    print(df_final.head(20).to_string())
    
    print("\n后20名球队:")
    print(df_final.tail(20).to_string())
    
    print("\n" + "="*80)
    print("统计信息:")
    print("="*80)
    print(f"总记录数: {len(df_final)}")
    print(f"有效TQ记录: {df_final['TQ'].notna().sum()}")
    print(f"有效胜率记录: {df_final['win_pct'].notna().sum()}")
    
    print("\nTQ统计:")
    print(df_final['TQ'].describe())
    
    print("\n胜率统计:")
    print(df_final['win_pct'].describe())
    
    # 8. 分析TQ与胜率的相关性
    if df_final['TQ'].notna().sum() > 0 and df_final['win_pct'].notna().sum() > 0:
        correlation = df_final[['TQ', 'win_pct']].corr().iloc[0, 1]
        print(f"\nTQ与胜率的相关系数: {correlation:.4f}")
        
        # 按年份分析
        print("\n各年份TQ与胜率相关系数:")
        for year in sorted(df_final['year'].unique()):
            year_data = df_final[df_final['year'] == year]
            if len(year_data) > 1:
                year_corr = year_data[['TQ', 'win_pct']].corr().iloc[0, 1]
                print(f"  {year}: {year_corr:.4f}")

def run_with_params(p_values: list, q_values: list):
    """
    使用不同的p、q参数运行计算
    
    参数:
        p_values: p参数列表
        q_values: q参数列表
    """
    for p in p_values:
        for q in q_values:
            print("\n" + "="*80)
            print(f"运行参数组合: p={p}, q={q}")
            print("="*80)
            main(p=p, q=q)
            print("\n")

if __name__ == "__main__":
    # 默认运行，使用p=2, q=2
    main(p=5.0, q=5.0)
    
    # 如果想要测试不同参数，可以使用：
    # run_with_params(p_values=[1.5, 2.0, 2.5], q_values=[1.5, 2.0, 2.5])
