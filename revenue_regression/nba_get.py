"""
从NBA API获取数据
生成两个文件：
1. nba_player_stat.csv: (player, team, year, attendance_rate)
2. nba_team_stat.csv: (team, year, wins, losses, win_pct)
"""

from nba_api.stats.static import teams
from nba_api.stats.endpoints import leaguedashplayerstats, leaguestandingsv3
import pandas as pd
import time
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_player_name(name: str) -> str:
    """清理球员名称，移除特殊字符"""
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

def get_team_name_by_id(team_id: int) -> str:
    """根据team_id获取球队全名"""
    all_teams = teams.get_teams()
    for team in all_teams:
        if team['id'] == team_id:
            return team['full_name']
    return "Unknown"

def get_season_player_stats(season: str) -> pd.DataFrame:
    """
    获取指定赛季所有球员的统计数据
    
    参数:
        season: 赛季，格式如 '2023-24'
    
    返回:
        包含球员统计的DataFrame
    """
    try:
        logging.info(f"正在获取 {season} 赛季球员数据...")
        time.sleep(0.6)  # API限流
        
        # 获取赛季所有球员统计
        player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season',
            per_mode_detailed='Totals'
        )
        
        df = player_stats.get_data_frames()[0]
        
        if df.empty:
            logging.warning(f"{season} 赛季没有球员数据")
            return pd.DataFrame()
        
        logging.info(f"获取到 {len(df)} 个球员的数据")
        return df
        
    except Exception as e:
        logging.error(f"获取 {season} 球员数据时出错: {str(e)}")
        return pd.DataFrame()

def get_season_team_standings(season: str) -> pd.DataFrame:
    """
    获取指定赛季球队战绩
    
    参数:
        season: 赛季，格式如 '2023-24'
    
    返回:
        包含球队战绩的DataFrame
    """
    try:
        logging.info(f"正在获取 {season} 赛季球队战绩...")
        time.sleep(0.6)  # API限流
        
        standings = leaguestandingsv3.LeagueStandingsV3(
            season=season,
            season_type='Regular Season'
        )
        
        df = standings.get_data_frames()[0]
        
        if df.empty:
            logging.warning(f"{season} 赛季没有球队战绩数据")
            return pd.DataFrame()
        
        logging.info(f"获取到 {len(df)} 支球队的战绩")
        return df
        
    except Exception as e:
        logging.error(f"获取 {season} 球队战绩时出错: {str(e)}")
        logging.info(f"尝试使用备用方法获取 {season} 赛季数据...")
        
        # 备用方法：使用TeamYearByYearStats
        try:
            from nba_api.stats.endpoints import teamyearbyyearstats
            all_teams = teams.get_teams()
            team_records = []
            
            for team in all_teams:
                time.sleep(0.6)
                try:
                    team_stats = teamyearbyyearstats.TeamYearByYearStats(
                        team_id=team['id'],
                        season_type_all_star='Regular Season'
                    )
                    df_team = team_stats.get_data_frames()[0]
                    season_data = df_team[df_team['YEAR'] == season]
                    
                    if not season_data.empty:
                        team_records.append({
                            'TeamID': team['id'],
                            'WINS': season_data['WINS'].values[0],
                            'LOSSES': season_data['LOSSES'].values[0]
                        })
                except Exception as team_error:
                    logging.warning(f"无法获取 {team['full_name']} 的数据: {str(team_error)}")
                    continue
            
            if team_records:
                logging.info(f"备用方法成功获取到 {len(team_records)} 支球队的战绩")
                return pd.DataFrame(team_records)
            else:
                return pd.DataFrame()
                
        except Exception as backup_error:
            logging.error(f"备用方法也失败了: {str(backup_error)}")
            return pd.DataFrame()

def process_player_data(years: list) -> pd.DataFrame:
    """
    处理所有年份的球员数据
    
    参数:
        years: 年份列表
    
    返回:
        球员统计DataFrame
    """
    all_player_data = []
    
    for year in years:
        # 转换年份格式：2024 -> 2023-24
        season_start = str(int(year) - 1)
        season_end = year[-2:]
        season = f"{season_start}-{season_end}"
        
        # 获取该赛季球员数据
        df = get_season_player_stats(season)
        
        if df.empty:
            continue
        
        # 提取需要的信息
        for _, row in df.iterrows():
            player_name = clean_player_name(row['PLAYER_NAME'])
            team_name = get_team_name_by_id(row['TEAM_ID'])
            games_played = row['GP']
            
            # 计算出勤率（需要知道球队总比赛数，常规赛通常是82场）
            # 但实际可能因停摆等原因不同，这里我们先用GP除以82的估算
            # 后面会用实际球队比赛数更新
            attendance_rate = games_played / 82.0
            
            all_player_data.append({
                'player': player_name,
                'team': team_name,
                'year': year,
                'games_played': games_played,
                'attendance_rate': attendance_rate  # 临时值，后续会更新
            })
    
    return pd.DataFrame(all_player_data)

def process_team_data(years: list) -> pd.DataFrame:
    """
    处理所有年份的球队战绩数据
    
    参数:
        years: 年份列表
    
    返回:
        球队战绩DataFrame
    """
    all_team_data = []
    
    for year in years:
        # 转换年份格式
        season_start = str(int(year) - 1)
        season_end = year[-2:]
        season = f"{season_start}-{season_end}"
        
        # 获取该赛季球队战绩
        df = get_season_team_standings(season)
        
        if df.empty:
            continue
        
        # 提取需要的信息
        for _, row in df.iterrows():
            team_name = get_team_name_by_id(row['TeamID'])
            wins = row['WINS']
            losses = row['LOSSES']
            total_games = wins + losses
            win_pct = wins / total_games if total_games > 0 else 0
            
            all_team_data.append({
                'team': team_name,
                'year': year,
                'wins': wins,
                'losses': losses,
                'total_games': total_games,
                'win_pct': win_pct
            })
    
    return pd.DataFrame(all_team_data)

def update_attendance_rates(df_player: pd.DataFrame, df_team: pd.DataFrame) -> pd.DataFrame:
    """
    用实际球队比赛数更新出勤率
    
    参数:
        df_player: 球员数据
        df_team: 球队数据
    
    返回:
        更新后的球员数据
    """
    # 合并获取实际球队比赛数
    df_merged = df_player.merge(
        df_team[['team', 'year', 'total_games']],
        on=['team', 'year'],
        how='left'
    )
    
    # 重新计算出勤率
    df_merged['attendance_rate'] = df_merged.apply(
        lambda row: row['games_played'] / row['total_games'] if row['total_games'] > 0 else 0,
        axis=1
    )
    
    # 只保留需要的列
    df_final = df_merged[['player', 'team', 'year', 'attendance_rate']]
    
    return df_final

def main():
    """主函数"""
    # 定义年份
    years = ['2015','2016', '2017', '2018','2019','2020', '2021', '2022', '2023', '2024','2025']
    
    # 检查RAPM文件是否存在（用于验证）
    rapm_files = [f'RAPM_CSV/{year}.csv' for year in years]
    existing_years = [year for year, f in zip(years, rapm_files) if os.path.exists(f)]
    
    if not existing_years:
        logging.error("未找到任何RAPM文件!")
        return
    
    logging.info(f"将处理以下年份: {existing_years}")
    
    # 1. 获取所有球员数据
    logging.info("\n" + "="*60)
    logging.info("步骤1: 获取球员统计数据")
    logging.info("="*60)
    df_player = process_player_data(existing_years)
    
    if df_player.empty:
        logging.error("无法获取球员数据!")
        return
    
    # 2. 获取所有球队战绩
    logging.info("\n" + "="*60)
    logging.info("步骤2: 获取球队战绩数据")
    logging.info("="*60)
    df_team = process_team_data(existing_years)
    
    if df_team.empty:
        logging.error("无法获取球队战绩!")
        return
    
    # 3. 更新出勤率
    logging.info("\n" + "="*60)
    logging.info("步骤3: 更新出勤率")
    logging.info("="*60)
    df_player_final = update_attendance_rates(df_player, df_team)
    
    # 4. 保存球员数据
    player_output = 'TQ_stat/nba_player_stat.csv'
    df_player_final.to_csv(player_output, index=False, encoding='utf-8')
    logging.info(f"\n球员数据已保存到: {player_output}")
    logging.info(f"总共 {len(df_player_final)} 条球员记录")
    
    # 5. 保存球队数据
    team_output = 'TQ_stat/nba_team_stat.csv'
    df_team.to_csv(team_output, index=False, encoding='utf-8')
    logging.info(f"球队数据已保存到: {team_output}")
    logging.info(f"总共 {len(df_team)} 条球队记录")
    
    # 6. 显示统计信息
    print("\n" + "="*60)
    print("球员数据预览 (前10行):")
    print("="*60)
    print(df_player_final.head(10))
    
    print("\n" + "="*60)
    print("球队数据预览:")
    print("="*60)
    print(df_team.head(10))
    
    print("\n" + "="*60)
    print("数据统计:")
    print("="*60)
    print(f"球员记录数: {len(df_player_final)}")
    print(f"球队记录数: {len(df_team)}")
    print(f"涵盖年份: {sorted(df_player_final['year'].unique())}")
    
    print("\n各年份球员数:")
    print(df_player_final['year'].value_counts().sort_index())
    
    print("\n各年份球队数:")
    print(df_team['year'].value_counts().sort_index())

if __name__ == "__main__":
    main()
