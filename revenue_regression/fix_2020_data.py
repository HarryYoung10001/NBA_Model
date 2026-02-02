"""
补充2019-20赛季的球队战绩数据
用于修复nba_get.py中未能获取的2019-20赛季数据
"""

from nba_api.stats.static import teams
from nba_api.stats.endpoints import teamyearbyyearstats
import pandas as pd
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_2019_20_team_standings():
    """获取2019-20赛季所有球队战绩"""
    all_teams = teams.get_teams()
    team_records = []
    season = '2019-20'
    
    logging.info(f"开始获取2019-20赛季所有球队战绩...")
    
    for idx, team in enumerate(all_teams, 1):
        try:
            time.sleep(0.6)  # API限流
            
            team_stats = teamyearbyyearstats.TeamYearByYearStats(
                team_id=team['id'],
                season_type_all_star='Regular Season'
            )
            
            df_team = team_stats.get_data_frames()[0]
            season_data = df_team[df_team['YEAR'] == season]
            
            if not season_data.empty:
                wins = season_data['WINS'].values[0]
                losses = season_data['LOSSES'].values[0]
                total_games = wins + losses
                win_pct = wins / total_games if total_games > 0 else 0
                
                team_records.append({
                    'team': team['full_name'],
                    'year': '2020',
                    'wins': wins,
                    'losses': losses,
                    'total_games': total_games,
                    'win_pct': win_pct
                })
                
                logging.info(f"[{idx}/{len(all_teams)}] {team['full_name']}: {wins}胜{losses}负")
            else:
                logging.warning(f"{team['full_name']} 没有2019-20赛季数据")
                
        except Exception as e:
            logging.error(f"获取 {team['full_name']} 数据时出错: {str(e)}")
            continue
    
    return pd.DataFrame(team_records)

def merge_with_existing_data(df_new: pd.DataFrame, existing_file: str):
    """将新数据合并到现有文件"""
    try:
        # 读取现有数据
        df_existing = pd.read_csv(existing_file)
        logging.info(f"现有数据: {len(df_existing)} 条记录")
        
        # 移除可能的重复数据（如果有的话）
        df_existing = df_existing[df_existing['year'] != '2020']
        
        # 合并
        df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        
        # 按年份排序
        df_merged = df_merged.sort_values(['year', 'team']).reset_index(drop=True)
        
        logging.info(f"合并后: {len(df_merged)} 条记录")
        
        return df_merged
        
    except Exception as e:
        logging.error(f"读取现有文件时出错: {str(e)}")
        return df_new

def main():
    """主函数"""
    # 1. 获取2019-20赛季数据
    df_2020 = get_2019_20_team_standings()
    
    if df_2020.empty:
        logging.error("无法获取2019-20赛季数据!")
        return
    
    logging.info(f"\n成功获取 {len(df_2020)} 支球队的2019-20赛季数据")
    
    # 2. 读取现有的nba_team_stat.csv并合并
    existing_file = '/mnt/user-data/outputs/nba_team_stat.csv'
    df_final = merge_with_existing_data(df_2020, existing_file)
    
    # 3. 保存更新后的数据
    output_file = '/mnt/user-data/outputs/nba_team_stat.csv'
    df_final.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"\n更新后的数据已保存到: {output_file}")
    
    # 4. 显示统计信息
    print("\n" + "="*60)
    print("更新后的球队战绩数据")
    print("="*60)
    print(f"总记录数: {len(df_final)}")
    print("\n各年份球队数:")
    print(df_final['year'].value_counts().sort_index())
    
    print("\n2019-20赛季数据预览:")
    print(df_final[df_final['year'] == '2020'].to_string())

if __name__ == "__main__":
    main()
