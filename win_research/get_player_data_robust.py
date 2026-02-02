"""
从nba-api获取球员数据（包括年龄）- 多种方法
方法1: 使用 commonallplayers 获取球员信息，然后匹配
方法2: 使用 commonplayerinfo 批量获取（优化版）
"""

from nba_api.stats.endpoints import leaguedashplayerstats, commonallplayers, commonplayerinfo
from nba_api.stats.static import players
import pandas as pd
import time
from datetime import datetime

def get_season_string(year):
    """将年份转换为赛季字符串，例如 2015 -> '2014-15'"""
    return f"{year-1}-{str(year)[2:]}"

def calculate_age_at_season_start(from_year, to_year, birth_year):
    """
    计算球员在赛季开始时的年龄
    from_year: 赛季开始年份（例如2014-15赛季是2014）
    to_year: 赛季结束年份（例如2014-15赛季是2015）
    birth_year: 出生年份
    """
    # 赛季通常在10月开始
    season_start_year = from_year
    return season_start_year - birth_year

# ============================================
# 方法1: 使用 commonallplayers (推荐)
# ============================================
def method1_commonallplayers():
    """
    使用 commonallplayers 端点
    这个端点返回所有球员的基本信息，包括出生年份
    只需要调用一次！
    """
    print("\n" + "=" * 70)
    print("方法1: 使用 commonallplayers 端点")
    print("=" * 70)
    
    try:
        # 获取所有球员的信息
        print("正在获取所有NBA球员的基本信息...")
        all_players_info = commonallplayers.CommonAllPlayers(
            is_only_current_season=0  # 获取历史所有球员
        )
        
        players_df = all_players_info.get_data_frames()[0]
        
        print(f"获取到 {len(players_df)} 名球员的信息")
        print("\n可用列:")
        print(players_df.columns.tolist())
        
        # 检查是否有 FROM_YEAR, TO_YEAR 或类似的字段
        # 通常包含: PERSON_ID, DISPLAY_FIRST_LAST, ROSTERSTATUS, FROM_YEAR, TO_YEAR, etc.
        
        # 创建球员ID到出生年份的映射（如果有的话）
        player_birth_map = {}
        if 'BIRTHDATE' in players_df.columns:
            for _, row in players_df.iterrows():
                player_name = row['DISPLAY_FIRST_LAST']
                birthdate = row['BIRTHDATE']
                if pd.notna(birthdate):
                    birth_year = pd.to_datetime(birthdate).year
                    player_birth_map[player_name] = birth_year
        
        print(f"\n示例数据:")
        print(players_df.head(10).to_string())
        
        return players_df, player_birth_map
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        return None, {}

# ============================================
# 方法2: 组合方法（最稳健）
# ============================================
def method2_combined_approach():
    """
    组合方法：
    1. 先获取每个赛季的球员统计（player, team, minutes）
    2. 使用 static players 获取球员ID和年龄
    3. 匹配合并
    """
    print("\n" + "=" * 70)
    print("方法2: 组合方法（推荐）")
    print("=" * 70)
    
    all_data = []
    
    # 使用 static players 获取所有球员信息
    print("\n获取球员静态信息...")
    all_players = players.get_players()
    # all_players 是一个字典列表，包含: id, full_name, is_active
    
    print(f"获取到 {len(all_players)} 名球员")
    print("示例数据:")
    print(all_players[:3])
    
    # 创建球员名到ID的映射
    player_name_to_id = {p['full_name']: p['id'] for p in all_players}
    
    # 获取每个赛季的数据
    for year in range(2015, 2025):
        season = get_season_string(year)
        print(f"\n正在获取 {season} 赛季数据...")
        
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = stats.get_data_frames()[0]
            
            # 基本字段
            df_selected = df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN']].copy()
            df_selected['year'] = year
            df_selected.columns = ['Player', 'Team', 'Minutes', 'year']
            
            print(f"  ✓ 获取到 {len(df_selected)} 名球员")
            all_data.append(df_selected)
            
            if year < 2024:
                time.sleep(1)
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    if not all_data:
        print("\n未能获取任何数据")
        return None
    
    # 合并所有赛季数据
    final_df = pd.concat(all_data, ignore_index=True)
    print(f"\n合并后总记录数: {len(final_df)}")
    
    # 现在需要添加年龄
    # 这里我们需要获取每个球员的出生年份
    print("\n正在获取球员年龄信息...")
    print("（这可能需要一些时间，因为需要逐个查询球员信息）")
    
    unique_players = final_df['Player'].unique()
    player_birth_years = {}
    
    # 批量获取，但使用缓存和延迟
    for i, player_name in enumerate(unique_players):
        if i % 50 == 0:
            print(f"  进度: {i}/{len(unique_players)}")
        
        if player_name in player_name_to_id:
            player_id = player_name_to_id[player_name]
            
            try:
                # 获取球员详细信息
                player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
                info_df = player_info.get_data_frames()[0]
                
                if len(info_df) > 0 and 'BIRTHDATE' in info_df.columns:
                    birthdate = info_df.iloc[0]['BIRTHDATE']
                    if pd.notna(birthdate):
                        birth_year = pd.to_datetime(birthdate).year
                        player_birth_years[player_name] = birth_year
                
                # 避免请求过快
                if i % 10 == 0 and i > 0:
                    time.sleep(0.5)
                    
            except Exception as e:
                continue
    
    print(f"\n成功获取 {len(player_birth_years)} 名球员的出生年份")
    
    # 计算年龄
    ages = []
    for _, row in final_df.iterrows():
        player_name = row['Player']
        year = row['year']
        
        if player_name in player_birth_years:
            birth_year = player_birth_years[player_name]
            # 赛季开始年份（year-1，例如2014-15赛季从2014年10月开始）
            age = (year - 1) - birth_year
            ages.append(age)
        else:
            ages.append(None)
    
    final_df['Age'] = ages
    
    # 保存
    output_file = 'person_time_with_age.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n" + "=" * 70)
    print("数据获取完成！")
    print("=" * 70)
    print(f"文件已保存: {output_file}")
    print(f"总记录数: {len(final_df)}")
    print(f"包含年龄的记录: {final_df['Age'].notna().sum()}")
    print(f"缺失年龄的记录: {final_df['Age'].isna().sum()}")
    
    return final_df

# ============================================
# 方法3: 简化方案（如果上述方法都失败）
# ============================================
def method3_simplified():
    """
    简化方案：先获取数据，年龄留空
    用户可以后续手动添加或使用其他数据源
    """
    print("\n" + "=" * 70)
    print("方法3: 简化方案（不包含年龄）")
    print("=" * 70)
    print("此方法只获取 player, team, minutes, year")
    print("年龄需要从其他来源获取")
    print("=" * 70)
    
    all_data = []
    
    for year in range(2015, 2025):
        season = get_season_string(year)
        print(f"\n正在获取 {season} 赛季数据...")
        
        try:
            stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                season_type_all_star='Regular Season'
            )
            
            df = stats.get_data_frames()[0]
            df_selected = df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN']].copy()
            df_selected['year'] = year
            df_selected.columns = ['Player', 'Team', 'Minutes', 'year']
            
            print(f"  ✓ 获取到 {len(df_selected)} 名球员")
            all_data.append(df_selected)
            
            if year < 2024:
                time.sleep(1)
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df['Age'] = None  # 年龄列留空
        
        output_file = 'person_time_no_age.csv'
        final_df.to_csv(output_file, index=False)
        
        print(f"\n文件已保存: {output_file}")
        print(f"总记录数: {len(final_df)}")
        print("⚠️  年龄列为空，需要从其他来源补充")
        
        return final_df
    
    return None

def main():
    """主函数：尝试不同的方法"""
    print("=" * 70)
    print("NBA球员数据获取工具")
    print("=" * 70)
    print("\n将尝试多种方法获取数据...")
    
    # 首先检查 commonallplayers
    print("\n尝试方法1...")
    players_df, birth_map = method1_commonallplayers()
    
    if players_df is not None:
        print("\n✓ 方法1成功，请查看返回的字段")
        print("如果有 FROM_YEAR 或 BIRTHDATE 等字段，可以用来计算年龄")
    
    # 如果方法1不够，尝试方法2
    print("\n" + "=" * 70)
    print("推荐: 使用方法2（组合方法）")
    print("=" * 70)
    print("\n如果您想继续，请运行:")
    print("  result = method2_combined_approach()")
    print("\n或者如果方法2太慢，使用方法3:")
    print("  result = method3_simplified()")

if __name__ == "__main__":
    main()
