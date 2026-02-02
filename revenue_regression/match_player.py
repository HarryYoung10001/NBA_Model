"""
匹配球员RAPM数据和NBA统计数据
输入: 5个RAPM文件(2020-2024.csv) 和 nba_stat.csv
输出: match_stat.csv，包含列(player, team, year, rapm_off, rapm_def, attendance_rate)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_rapm_value(value: str) -> float:
    """
    解析RAPM值，提取数值部分
    例如: "6.1 (99)" -> 6.1
    """
    if pd.isna(value):
        return np.nan
    
    # 移除括号及其内容
    match = re.match(r'([-\d.]+)', str(value).strip())
    if match:
        return float(match.group(1))
    return np.nan

def clean_player_name(name: str) -> str:
    """清理球员名称，标准化格式"""
    # 替换特殊字符
    replacements = {
        '?': 'c',
        '??': 'ng',
        '\xa8\xa6': 'e',
        'ø': 'o',
        'ć': 'c',
        'č': 'c',
        'š': 's',
        'ž': 'z',
    }
    
    cleaned = str(name)
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    
    return cleaned.strip()

def load_rapm_data(file_path: str, year: str) -> pd.DataFrame:
    """
    加载单个RAPM文件
    
    参数:
        file_path: RAPM文件路径
        year: 年份
    
    返回:
        包含球员RAPM数据的DataFrame
    """
    # 尝试多种编码格式
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-8-sig']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            logging.info(f"成功使用 {encoding} 编码读取 {file_path}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logging.error(f"使用 {encoding} 编码读取 {file_path} 时出错: {str(e)}")
            continue
    
    if df is None:
        logging.error(f"无法使用任何编码读取 {file_path}")
        return pd.DataFrame()
    
    try:
        # 检查必要的列是否存在
        required_cols = ['Player', 'Offense', 'Defense(*)']
        if not all(col in df.columns for col in required_cols):
            # 尝试其他可能的列名
            if 'Defense' in df.columns:
                df = df.rename(columns={'Defense': 'Defense(*)'})
            else:
                logging.error(f"{file_path} 缺少必要的列")
                return pd.DataFrame()
        
        # 清理球员名称
        df['player'] = df['Player'].apply(clean_player_name)
        
        # 解析RAPM数值
        df['rapm_off'] = df['Offense'].apply(parse_rapm_value)
        df['rapm_def'] = df['Defense(*)'].apply(parse_rapm_value)
        
        # 添加年份
        df['year'] = year
        
        # 只保留需要的列
        df = df[['player', 'year', 'rapm_off', 'rapm_def']]
        
        # 移除缺失值
        df = df.dropna(subset=['rapm_off', 'rapm_def'])
        
        logging.info(f"从 {file_path} 加载了 {len(df)} 个球员的数据")
        
        return df
        
    except Exception as e:
        logging.error(f"处理 {file_path} 时出错: {str(e)}")
        return pd.DataFrame()

def load_all_rapm_data(years: list) -> pd.DataFrame:
    """
    加载所有年份的RAPM数据
    
    参数:
        years: 年份列表
    
    返回:
        合并后的RAPM数据DataFrame
    """
    all_rapm = []
    
    # 尝试多个可能的路径
    possible_paths = [
        'RAPM_CSV/{year}.csv',  # Windows当前目录
        '/mnt/project/{year}.csv',  # Linux挂载目录
        '{year}.csv',  # 当前目录
        'data/{year}.csv',  # data子目录
    ]
    
    for year in years:
        file_loaded = False
        
        for path_template in possible_paths:
            file_path = path_template.format(year=year)
            
            # 检查文件是否存在
            import os
            if not os.path.exists(file_path):
                continue
            
            df = load_rapm_data(file_path, year)
            if not df.empty:
                all_rapm.append(df)
                file_loaded = True
                break
        
        if not file_loaded:
            logging.warning(f"未能加载 {year} 年的RAPM数据，尝试的路径: {[p.format(year=year) for p in possible_paths]}")
    
    if not all_rapm:
        logging.error("未能加载任何RAPM数据!")
        return pd.DataFrame()
    
    combined = pd.concat(all_rapm, ignore_index=True)
    logging.info(f"总共加载了 {len(combined)} 条RAPM记录")
    
    return combined

def load_nba_stats(file_path: str = None) -> pd.DataFrame:
    """
    加载NBA统计数据
    
    参数:
        file_path: nba_player_stat.csv文件路径
    
    返回:
        NBA统计数据DataFrame
    """
    # 尝试多个可能的路径
    if file_path is None:
        possible_paths = [
            'TQ_stat/nba_player_stat.csv',  # Windows用户指定的路径
            '/mnt/user-data/outputs/nba_player_stat.csv',  # Linux输出路径
            'nba_player_stat.csv',  # 当前目录
            'output/nba_player_stat.csv',  # output子目录
        ]
    else:
        possible_paths = [file_path]
    
    import os
    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                
                # 清理球员名称
                df['player'] = df['player'].apply(clean_player_name)
                
                # 转换年份为字符串
                df['year'] = df['year'].astype(str)
                
                logging.info(f"从 {path} 加载了 {len(df)} 条NBA统计记录")
                
                return df
                
            except Exception as e:
                logging.error(f"加载 {path} 时出错: {str(e)}")
                continue
    
    logging.error(f"无法找到NBA统计文件，尝试的路径: {possible_paths}")
    return pd.DataFrame()

def match_data(df_rapm: pd.DataFrame, df_nba: pd.DataFrame) -> pd.DataFrame:
    """
    匹配RAPM数据和NBA统计数据
    
    参数:
        df_rapm: RAPM数据
        df_nba: NBA统计数据
    
    返回:
        匹配后的数据DataFrame
    """
    # 基于player和year进行匹配
    df_matched = pd.merge(
        df_rapm,
        df_nba[['player', 'year', 'team', 'attendance_rate']],
        on=['player', 'year'],
        how='left'
    )
    
    # 统计匹配情况
    total_records = len(df_matched)
    matched_records = df_matched['team'].notna().sum()
    unmatched_records = total_records - matched_records
    
    logging.info(f"\n匹配统计:")
    logging.info(f"- 总记录数: {total_records}")
    logging.info(f"- 成功匹配: {matched_records} ({matched_records/total_records*100:.1f}%)")
    logging.info(f"- 未匹配: {unmatched_records} ({unmatched_records/total_records*100:.1f}%)")
    
    # 显示部分未匹配的球员
    if unmatched_records > 0:
        unmatched = df_matched[df_matched['team'].isna()][['player', 'year']].head(10)
        logging.warning(f"\n部分未匹配的球员示例:\n{unmatched}")
    
    return df_matched

def clean_matched_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清理匹配后的数据，处理缺失值
    
    参数:
        df: 匹配后的数据
    
    返回:
        清理后的数据
    """
    # 只保留有球队信息的记录
    df_clean = df[df['team'].notna()].copy()
    
    # 对于缺失的attendance_rate，设置为0或其他默认值
    # 这里我们用0表示该球员该赛季没有出场
    df_clean['attendance_rate'] = df_clean['attendance_rate'].fillna(0)
    
    # 选择最终的列，按指定顺序
    df_final = df_clean[['player', 'team', 'year', 'rapm_off', 'rapm_def', 'attendance_rate']]
    
    # 按年份和球队排序
    df_final = df_final.sort_values(['year', 'team', 'player']).reset_index(drop=True)
    
    logging.info(f"清理后保留 {len(df_final)} 条有效记录")
    
    return df_final

def main():
    """主函数"""
    # 定义年份
    years = ['2015','2016','2017','2018','2019','2020', '2021', '2022', '2023', '2024']
    
    # 1. 加载所有RAPM数据
    logging.info("步骤1: 加载RAPM数据...")
    df_rapm = load_all_rapm_data(years)
    
    if df_rapm.empty:
        logging.error("无法继续，因为没有RAPM数据")
        return
    
    # 2. 加载NBA统计数据
    logging.info("\n步骤2: 加载NBA统计数据...")
    df_nba = load_nba_stats()
    
    if df_nba.empty:
        logging.error("无法继续，因为没有NBA统计数据")
        logging.info("请先运行 nba_get.py 生成 nba_stat.csv")
        return
    
    # 3. 匹配数据
    logging.info("\n步骤3: 匹配数据...")
    df_matched = match_data(df_rapm, df_nba)
    
    # 4. 清理数据
    logging.info("\n步骤4: 清理数据...")
    df_final = clean_matched_data(df_matched)
    
    # 5. 保存结果
    import os
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
    
    output_file = os.path.join(output_dir, 'match_stat.csv')
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"\n数据已保存到 {output_file}")
    
    # 6. 显示统计信息
    print("\n" + "="*60)
    print("数据预览 (前10行):")
    print("="*60)
    print(df_final.head(10).to_string())
    
    print("\n" + "="*60)
    print("数据统计:")
    print("="*60)
    print(f"总记录数: {len(df_final)}")
    print(f"涵盖年份: {sorted(df_final['year'].unique())}")
    print(f"涵盖球队数: {df_final['team'].nunique()}")
    print(f"涵盖球员数: {df_final['player'].nunique()}")
    
    print("\n各年份记录数:")
    print(df_final['year'].value_counts().sort_index())
    
    print("\n各球队记录数 (Top 10):")
    print(df_final['team'].value_counts().head(10))
    
    print("\nRAPM统计:")
    print(df_final[['rapm_off', 'rapm_def', 'attendance_rate']].describe())

if __name__ == "__main__":
    main()
