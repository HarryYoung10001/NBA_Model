import pandas as pd
import numpy as np

# 读取原始数据
input_file = 'csv_fold/Merged_NBA_Player_Stats_2015_2025.csv'
output_file = 'csv_fold/AGE_USG%.csv'

# 读取CSV文件
df = pd.read_csv(input_file)

# 存储所有数据点的列表
age_performance_data = []

# 定义赛季列表（从列名推断）
seasons = [
    '2015-2016', '2016-2017', '2017-2018', '2018-2019', '2019-2020',
    '2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025'
]

# 遍历每个球员（每一行）
for idx, row in df.iterrows():
    # 遍历每个赛季
    for season in seasons:
        age_col = f'{season}Age'
        usg_col = f'{season}USG%'
        
        # 检查这两列是否存在
        if age_col in df.columns and usg_col in df.columns:
            age = row[age_col]
            usg = row[usg_col]
            
            # 只有当年龄和使用率都不为空时才添加数据点
            if pd.notna(age) and pd.notna(usg):
                age_performance_data.append({
                    'AGE': age,
                    'Performance': usg
                })

# 创建新的DataFrame
result_df = pd.DataFrame(age_performance_data)
# 先按AGE排序，再按Performance排序
df_sorted = result_df.sort_values(by=['AGE', 'Performance'], ascending=[True, True])

# 保存到新的CSV文件
df_sorted.to_csv(output_file, index=False)

# 打印统计信息
print(f"数据处理完成！")
print(f"总共提取了 {len(result_df)} 个数据点")
print(f"\n数据摘要：")
print(result_df.describe())
print(f"\n前10行数据预览：")
print(result_df.head(10))
print(f"\n数据已保存至: {output_file}")

