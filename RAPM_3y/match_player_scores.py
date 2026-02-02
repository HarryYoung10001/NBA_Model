import pandas as pd
import numpy as np

# 读取三个输入文件
print("正在读取文件...")
player_scores = pd.read_csv('csv_fold/player_scores.csv')
commercial_value = pd.read_csv('csv_fold/league_commercial_score.csv')

# 🔧 方法2：正常读取CSV（保留header），然后选择T列
potential_results = pd.read_csv('csv_fold/potential_results.csv')

print(f"球员得分数据: {len(player_scores)} 行")
print(f"商业价值数据: {len(commercial_value)} 行")
print(f"积分结果数据: {len(potential_results)} 行")

print("\n列名：", list(potential_results.columns))
print("\n前5行potential数据预览：")
print(potential_results.head())

# 创建商业价值字典
commercial_dict = dict(zip(commercial_value['name'], 
                          commercial_value['final_commercial_score']))

# 🔧 使用T列创建映射
age_to_potential = {}
for idx, row in potential_results.iterrows():
    if pd.notna(row['AGE']) and pd.notna(row['T']):
        age_to_potential[float(row['AGE'])] = float(row['T'])

print(f"\n有效的年龄-潜力映射: {len(age_to_potential)} 条")
print("前10个映射（T列的值）：")
for i, (age, t_val) in enumerate(list(age_to_potential.items())[:10]):
    print(f"  年龄 {age:.0f} -> T值 {t_val:.6f}")

# 准备结果列表
results = []

print("\n开始处理球员数据...")
processed_count = 0
missing_age_count = 0  # 统计缺失年龄的数量

for idx, row in player_scores.iterrows():
    player = row['Player']
    team = row['Team']
    age = row['AGE']
    athletic_score = row['Athletic_Score']
    
    final_commercial_score = commercial_dict.get(player, 0)
    
    # 查找potential（T值）
    if pd.isna(age):
        potential = 1.0  # 🔧 修改：无年龄时potential=1.0
        missing_age_count += 1
    else:
        try:
            age_float = float(age)
            
            if age_float in age_to_potential:
                potential = float(age_to_potential[age_float])
            else:
                available_ages = list(age_to_potential.keys())
                if available_ages:
                    closest_age = min(available_ages, key=lambda x: abs(float(x) - age_float))
                    potential = float(age_to_potential[closest_age])
                    if processed_count < 5:
                        print(f"  年龄 {age_float} -> 使用年龄 {closest_age} 的T值 {potential:.6f}")
                        processed_count += 1
                else:
                    potential = 1.0
        except (ValueError, TypeError):
            potential = 1.0  # 🔧 修改：年龄转换失败时potential=1.0
            missing_age_count += 1
    
    results.append({
        'Player': player,
        'Team': team,
        'AGE': age,
        'Athletic_Score': athletic_score,
        'final_commercial_score': final_commercial_score,
        'potential': potential
    })

# 创建结果DataFrame
result_df = pd.DataFrame(results)

# 确保数值类型
result_df['AGE'] = pd.to_numeric(result_df['AGE'], errors='coerce')
result_df['Athletic_Score'] = pd.to_numeric(result_df['Athletic_Score'], errors='coerce')
result_df['final_commercial_score'] = pd.to_numeric(result_df['final_commercial_score'], errors='coerce')
result_df['potential'] = pd.to_numeric(result_df['potential'], errors='coerce')
result_df = result_df.fillna(0)

# 保存
output_file = 'csv_fold/3scores.csv'
result_df.to_csv(output_file, index=False)

print(f"\n处理完成！共处理 {len(result_df)} 名球员")
print(f"缺失年龄的球员数（potential设为1.0）: {missing_age_count}")
print(f"结果已保存到: {output_file}")

print("\n前10行结果预览：")
print(result_df.head(10).to_string())

print(f"\n统计信息：")
print(f"有商业价值数据的球员数: {(result_df['final_commercial_score'] > 0).sum()}")
print(f"T值范围: {result_df['potential'].min():.6f} - {result_df['potential'].max():.6f}")
print(f"T值=1.0的球员数: {(result_df['potential'] == 1.0).sum()}")
print(f"\nT值分布统计：")
print(result_df['potential'].describe())

print("\n不同年龄的T值示例：")
for age in [20, 25, 30, 35, 40]:
    age_players = result_df[result_df['AGE'] == age]
    if len(age_players) > 0:
        print(f"  年龄{age}: {age_players.iloc[0]['Player']}, T值={age_players.iloc[0]['potential']:.6f}")

# 显示一些无年龄的球员示例
no_age_players = result_df[result_df['AGE'] == 0]
if len(no_age_players) > 0:
    print(f"\n无年龄数据的球员示例（potential=1.0）：")
    for i in range(min(5, len(no_age_players))):
        player_row = no_age_players.iloc[i]
        print(f"  {player_row['Player']}, Team={player_row['Team']}, T值={player_row['potential']:.6f}")

