import pandas as pd
import numpy as np
from math import sqrt
from scipy.special import erf

def calculate_team_quality(input_file, output_file, p=3, q=3):
    """
    计算每个队伍的质量指标
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        p: RAPM_offense的幂次（奇正整数）
        q: RAPM_defense的幂次（奇正整数）
    
    公式:
        Team_Quality = (Σ Attendance_i · (RAPM^off_i)^p)^(1/p) - (Σ Attendance_i · (RAPM^def_i)^q)^(1/q)
    """
    
    # 读取数据
    print(f"正在读取文件: {input_file}")
    df = pd.read_csv(input_file)
    
    # 确保p和q是奇正整数
    assert p > 0 and p % 2 == 1, "p必须是奇正整数"
    assert q > 0 and q % 2 == 1, "q必须是奇正整数"
    
    print(f"数据包含 {len(df)} 名球员，来自 {df['Team'].nunique()} 支球队")
    print(f"使用参数: p={p}, q={q}")
    
    # 按队伍分组计算
    team_quality_list = []
    
    for team in df['Team'].unique():
        # 跳过缺失值
        if pd.isna(team):
            continue
            
        # 获取该队伍的所有球员数据
        team_data = df[df['Team'] == team].copy()
        
        # 计算进攻部分: (Σ Attendance_i · (RAPM^off_i)^p)^(1/p)
        # 由于p是奇数，可以直接计算带符号的幂次
        offense_sum = (team_data['Attendance'] * np.power(team_data['RAPM_offense'], p)).sum()
        offense_component = np.sign(offense_sum) * np.power(np.abs(offense_sum), 1/p)
        
        # 计算防守部分: (Σ Attendance_i · (RAPM^def_i)^q)^(1/q)
        # 由于q是奇数，可以直接计算带符号的幂次
        defense_sum = (team_data['Attendance'] * np.power(team_data['RAPM_defense'], q)).sum()
        defense_component = np.sign(defense_sum) * np.power(np.abs(defense_sum), 1/q)
        
        # 计算Team_Quality
        team_quality = offense_component - defense_component
        
        team_quality_list.append({
            'Team': team,
            'Quality': team_quality,
            'Player_Count': len(team_data)
        })
    
    # 创建结果DataFrame并按Quality降序排序
    result_df = pd.DataFrame(team_quality_list)
    result_df = result_df.sort_values('Quality', ascending=False).reset_index(drop=True)
    
        # 创建结果DataFrame并按Quality降序排序
    result_df = pd.DataFrame(team_quality_list)
    result_df = result_df.sort_values('Quality', ascending=False).reset_index(drop=True)

    # === 新增：基于 Quality 的 z-score ===
    # ddof=0: 总体标准差；ddof=1: 样本标准差（可按需要改）
    ddof = 0
    quality_mean = result_df['Quality'].mean()
    quality_std = result_df['Quality'].std(ddof=ddof)

    # 避免所有球队 Quality 完全一样导致 std=0 的情况
    if quality_std == 0 or np.isnan(quality_std):
        result_df['Quality_Z'] = 0.0
    else:
        result_df['Quality_Z'] = (result_df['Quality'] - quality_mean) / quality_std
        
    result_df["Normalized_Quality"] = 0.5 * (1.0 + erf(result_df["Quality_Z"] / sqrt(2.0)))
        
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    print(f"\n前5名队伍:")
    print(result_df.head(10).to_string(index=False))
    
    return result_df


if __name__ == "__main__":
    # 设置输入输出文件路径
    input_file = "csv_fold/player_scores.csv"
    output_file = "csv_fold/team_quality.csv"
    
    # 计算队伍质量，p=3, q=3（可以根据需要修改）
    result = calculate_team_quality(input_file, output_file, p=3, q=3)
    
    print(f"\n总计 {len(result)} 支队伍")

