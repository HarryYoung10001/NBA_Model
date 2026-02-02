import pandas as pd
import numpy as np
from scipy.special import erf

# 读取CSV文件
df = pd.read_csv('TQ_stat/TQ_stat.csv')

print("原始数据列名:", df.columns.tolist())
print("原始数据形状:", df.shape)

# 按年份分组计算Z-score
def calculate_zscore_by_year(df):
    """
    按年份计算每个队伍的TQ Z-score
    """
    # 创建新列存储Z-score
    df['TQ_Z_score'] = np.nan
    
    # 对每个年份分别计算
    for year in df['year'].unique():
        # 获取该年份的数据索引
        year_mask = df['year'] == year
        
        # 计算该年份的均值和标准差
        year_tq = df.loc[year_mask, 'TQ']
        mean_tq = year_tq.mean()
        std_tq = year_tq.std()
        
        # 计算Z-score
        df.loc[year_mask, 'TQ_Z_score'] = (year_tq - mean_tq) / std_tq
        df.loc[year_mask,'Normalized_TQ_Z_score'] = 0.5 * (1.0 + erf((year_tq - mean_tq) / std_tq))
    
    return df

# 计算Z-score
df = calculate_zscore_by_year(df)

print("\n处理后数据列名:", df.columns.tolist())
print("处理后数据形状:", df.shape)

# 保存结果到新文件
output_path = 'TQ_stat/TQ_stat_with_zscore.csv'
df.to_csv(output_path, index=False)

# 打印前几行查看结果
print("\n处理完成！前10行数据：")
print(df.head(10))

# 验证每年都有数据
print("\n每年的数据行数：")
print(df.groupby('year').size())

print(f"\n结果已保存到: {output_path}")
    # 8. 分析TQ与胜率的相关性
if df['TQ_Z_score'].notna().sum() > 0 and df['win_pct'].notna().sum() > 0:
    correlation = df[['TQ_Z_score', 'win_pct']].corr().iloc[0, 1]
    print(f"\nTQ与胜率的相关系数: {correlation:.4f}")
        
    # 按年份分析
    print("\n各年份TQ与胜率相关系数:")
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        if len(year_data) > 1:
            year_corr = year_data[['TQ_Z_score', 'win_pct']].corr().iloc[0, 1]
            print(f"  {year}: {year_corr:.4f}")
