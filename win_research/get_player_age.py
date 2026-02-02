"""
从nba-api获取球员数据（包括年龄）
每个赛季调用一次API，获取所有球员的 (player, team, time, age, year)
"""

from nba_api.stats.endpoints import leaguedashplayerstats
import pandas as pd
import time

def get_season_string(year):
    """将年份转换为赛季字符串，例如 2015 -> '2014-15'"""
    return f"{year-1}-{str(year)[2:]}"

def get_player_stats_for_season(year):
    """
    获取指定赛季的所有球员统计数据
    一次调用获取该赛季所有球员的：姓名、球队、出场时间、年龄
    """
    season = get_season_string(year)
    print(f"正在获取 {season} 赛季数据...")
    
    try:
        # 获取该赛季的球员统计数据
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season'
        )
        
        df = stats.get_data_frames()[0]
        
        # 选择需要的列
        # PLAYER_NAME = 球员姓名
        # TEAM_ABBREVIATION = 球队缩写
        # MIN = 总出场分钟数
        # PLAYER_AGE = 球员年龄（赛季开始时的年龄）
        columns_needed = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN', 'PLAYER_AGE']
        
        df_selected = df[columns_needed].copy()
        df_selected['year'] = year
        
        # 重命名列
        df_selected.columns = ['Player', 'Team', 'Minutes', 'Age', 'year']
        
        print(f"  成功获取 {len(df_selected)} 名球员的数据")
        return df_selected
    
    except Exception as e:
        print(f"  ❌ 获取 {season} 赛季数据时出错: {e}")
        return None

def main():
    """主函数：获取2015-2024赛季的数据"""
    all_data = []
    
    print("=" * 70)
    print("从nba-api获取球员数据（包括年龄）")
    print("=" * 70)
    print(f"\n将获取 2015-2024 赛季的数据...")
    print("每个赛季调用一次API，总共10次调用\n")
    
    # 获取2015-2024赛季的数据
    for year in range(2015, 2025):
        df = get_player_stats_for_season(year)
        
        if df is not None:
            all_data.append(df)
            print(f"  ✓ {year} 赛季: {len(df)} 名球员")
        else:
            print(f"  ✗ {year} 赛季: 获取失败")
        
        # 避免请求过快（除了最后一次）
        if year < 2024:
            print("  等待1秒...")
            time.sleep(1)
    
    # 合并所有数据
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 保存为CSV
        output_file = 'person_time_with_age.csv'
        final_df.to_csv(output_file, index=False)
        
        print("\n" + "=" * 70)
        print("数据获取完成！")
        print("=" * 70)
        print(f"\n文件已保存: {output_file}")
        print(f"总记录数: {len(final_df)}")
        print(f"唯一球员数: {final_df['Player'].nunique()}")
        print(f"赛季数: {final_df['year'].nunique()}")
        
        # 显示统计信息
        print("\n按年份统计:")
        year_stats = final_df.groupby('year').size()
        for year, count in year_stats.items():
            print(f"  {year}: {count} 名球员")
        
        print("\n年龄分布:")
        print(f"  最小年龄: {final_df['Age'].min()}")
        print(f"  最大年龄: {final_df['Age'].max()}")
        print(f"  平均年龄: {final_df['Age'].mean():.2f}")
        
        print("\n前10行数据预览:")
        print(final_df.head(10).to_string(index=False))
        
        # 检查缺失值
        missing = final_df.isnull().sum()
        if missing.any():
            print("\n⚠️  缺失值统计:")
            print(missing[missing > 0])
        else:
            print("\n✓ 无缺失值")
        
        return final_df
    else:
        print("\n❌ 未能获取任何数据")
        return None

if __name__ == "__main__":
    main()

