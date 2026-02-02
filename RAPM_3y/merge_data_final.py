import pandas as pd
import numpy as np

# 读取三个CSV文件
potential_results = pd.read_csv('csv_fold/potential_results.csv')
commercial_score = pd.read_csv('csv_fold/league_commercial_score.csv')
athletic_salary = pd.read_csv('csv_fold/athetic-salary.csv')

# 创建年龄到Normalized_T的映射
age_to_pw = dict(zip(potential_results['AGE'], potential_results['Normalized_T']))

# 创建球员名到commercial_score的映射
commercial_dict = {}
for _, row in commercial_score.iterrows():
    player_name = str(row['name']).strip()
    commercial_dict[player_name] = {
        'team': str(row['team']).strip() if pd.notna(row['team']) else "",
        'final_commercial_score': row['final_commercial_score'] if pd.notna(row['final_commercial_score']) else 0
    }

# 处理athletic_salary数据（以此为主表）
result_data = []

for _, row in athletic_salary.iterrows():
    player_name = str(row['Player']).strip()
    team = str(row['Team']).strip() if pd.notna(row['Team']) else ""
    age = row['AGE'] if pd.notna(row['AGE']) else 0
    athletic_score = row['Athletic_Score'] if pd.notna(row['Athletic_Score']) else 0
    attendance = row['Attendance'] if pd.notna(row['Attendance']) else 0
    salary = row['Salary'] if pd.notna(row['Salary']) else 0
    
    # 根据年龄获取PW值
    if age in age_to_pw:
        pw = age_to_pw[age]
    else:
        # 如果年龄不在映射表中，尝试最接近的年龄或设为0
        if age > 0 and age < 50:  # 合理的年龄范围
            # 找最接近的年龄
            ages = list(age_to_pw.keys())
            closest_age = min(ages, key=lambda x: abs(x - age))
            pw = age_to_pw[closest_age]
        else:
            pw = 0
    
    # 尝试匹配commercial数据
    final_commercial_score = 0
    commercial_team = team  # 默认使用athletic表中的team
    
    # 先尝试直接匹配
    if player_name in commercial_dict:
        final_commercial_score = commercial_dict[player_name]['final_commercial_score']
        if commercial_dict[player_name]['team']:
            commercial_team = commercial_dict[player_name]['team']
    else:
        # 尝试反转名字格式（"姓 名" <-> "名 姓"）
        parts = player_name.split()
        if len(parts) >= 2:
            reversed_name = ' '.join(parts[::-1])
            if reversed_name in commercial_dict:
                final_commercial_score = commercial_dict[reversed_name]['final_commercial_score']
                if commercial_dict[reversed_name]['team']:
                    commercial_team = commercial_dict[reversed_name]['team']
    
    result_data.append({
        'player_name': player_name,
        'team': commercial_team,
        'age': age,
        'PW': pw,
        'final_commercial_score': final_commercial_score,
        'athletic_score': athletic_score,
        'attendance_rate': attendance,
        'salary': salary
    })

# 创建结果DataFrame
result_df = pd.DataFrame(result_data)

# 保存到CSV
output_path = 'csv_fold/merged_player_data.csv'
result_df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"数据合并完成！")
print(f"总共处理了 {len(result_df)} 名球员")
print(f"\n前20行预览：")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(result_df.head(20).to_string())
print(f"\n数据统计：")
print(f"有年龄数据的球员: {(result_df['age'] > 0).sum()}")
print(f"有PW数据的球员: {(result_df['PW'] > 0).sum()}")
print(f"有商业分数的球员: {(result_df['final_commercial_score'] > 0).sum()}")
print(f"有体能分数的球员: {(result_df['athletic_score'] > 0).sum()}")
print(f"有出勤率数据的球员: {(result_df['attendance_rate'] > 0).sum()}")
print(f"有薪资数据的球员: {(result_df['salary'] > 0).sum()}")
print(f"\n缺失数据统计：")
print(f"缺失年龄的球员: {(result_df['age'] == 0).sum()}")
print(f"缺失PW的球员: {(result_df['PW'] == 0).sum()}")
print(f"缺失商业分数的球员: {(result_df['final_commercial_score'] == 0).sum()}")
print(f"缺失体能分数的球员: {(result_df['athletic_score'] == 0).sum()}")
print(f"缺失出勤率的球员: {(result_df['attendance_rate'] == 0).sum()}")
print(f"缺失薪资的球员: {(result_df['salary'] == 0).sum()}")
print(f"\n薪资统计：")
print(f"最高薪资: ${result_df['salary'].max():,.0f}")
print(f"最低薪资（非0）: ${result_df[result_df['salary'] > 0]['salary'].min():,.0f}")
print(f"平均薪资（非0）: ${result_df[result_df['salary'] > 0]['salary'].mean():,.0f}")
print(f"薪资中位数（非0）: ${result_df[result_df['salary'] > 0]['salary'].median():,.0f}")
