#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从player_scores.csv中提取AGE和Performance数据
"""

import pandas as pd

# 读取原始数据文件
df = pd.read_csv('csv_fold/player_scores.csv')

# 提取AGE和Performance列
df_extracted = df[['AGE', 'Performance']].copy()

# 删除缺失值
df_clean = df_extracted.dropna()
# 先按AGE排序，再按Performance排序
df_sorted = df_clean.sort_values(by=['AGE', 'Performance'], ascending=[True, True])

# 保存到新的CSV文件
df_sorted.to_csv('csv_fold/AGE-Performance.csv', index=False)

print(f"数据提取完成！")
print(f"原始数据行数: {len(df)}")
print(f"提取后数据行数: {len(df_extracted)}")
print(f"删除缺失值后行数: {len(df_clean)}")
print(f"已保存到: AGE-Performance.csv")

