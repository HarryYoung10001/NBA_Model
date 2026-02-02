"""
NBA球员数据获取脚本（改进版 - 包含年龄）
功能：
1. 从本地HTML文件读取RAPM数据
2. 使用nba-api获取球员出勤率和统计数据（包含年龄）
3. 清理和合并数据并保存为CSV
"""

import time
import pandas as pd
import requests
from typing import Optional
from io import StringIO
import re


# ============ 第一部分：获取RAPM数据 ============

def fetch_xrapm_table(html_file: str = "RAPM_3y.html") -> pd.DataFrame:
    """从本地HTML文件获取RAPM数据"""
    print(f"正在从本地文件读取RAPM数据: {html_file}")
    
    try:
        # 读取本地HTML文件
        with open(html_file, "r", encoding="utf-8") as f:
            html = f.read()
        
        print(f"✓ HTML文件读取成功，长度: {len(html)}")
        
        # 使用StringIO避免FutureWarning
        tables = pd.read_html(StringIO(html))
        print(f"找到 {len(tables)} 个表格")

        # 只打印表格的基本信息
        for i, t in enumerate(tables):
            print(f"表格 {i+1}: 形状={t.shape}, 列数={len(t.columns)}")

        # 选出"像 xRAPM 榜单"的表：列齐全、行数>0
        need_cols = {"Player", "Team", "Offense", "Total"}
        picked = None
        for i, t in enumerate(tables):
            cols = {str(c).strip() for c in t.columns}
            if need_cols.issubset(cols) and len(t) > 0:
                picked = t.copy()
                print(f"✓ 选中表格 {i+1}")
                break

        if picked is None:
            raise RuntimeError(
                f"未找到预期的xRAPM表格。\n"
                f"找到 {len(tables)} 个表格，但没有匹配的列结构。\n"
                f"需要的列: {need_cols}"
            )

        # 标准化列名
        picked.columns = [str(c).strip() for c in picked.columns]
        
        # 清理数据：提取数值部分（去除百分位数）
        print("正在清理RAPM数据...")
        for col in ['Offense', 'Defense(*)', 'Total']:
            if col in picked.columns:
                # 提取数值部分，如 "4.6 (99)" -> 4.6
                picked[col + '_value'] = picked[col].apply(
                    lambda x: float(re.search(r'(-?\d+\.?\d*)', str(x)).group(1)) 
                    if pd.notna(x) and re.search(r'(-?\d+\.?\d*)', str(x)) else None
                )
                # 提取百分位数部分，如 "4.6 (99)" -> 99
                picked[col + '_percentile'] = picked[col].apply(
                    lambda x: int(re.search(r'\((\d+)\)', str(x)).group(1))
                    if pd.notna(x) and re.search(r'\((\d+)\)', str(x)) else None
                )
        
        # 重命名Defense列
        if 'Defense(*)' in picked.columns:
            picked = picked.rename(columns={'Defense(*)': 'Defense'})
        
        print(f"✓ 成功获取RAPM数据: {picked.shape[0]} 名球员")
        print(f"✓ 数据清理完成，新增数值和百分位数列")
        return picked
        
    except FileNotFoundError:
        raise FileNotFoundError(
            f"错误：未找到文件 '{html_file}'\n"
            f"请确保文件与脚本在同一目录下，或提供正确的文件路径。"
        )
    except Exception as e:
        raise RuntimeError(f"解析HTML文件时出错: {type(e).__name__}: {e}")


# ============ 第二部分：使用nba-api获取出勤率数据 ============

def fetch_player_stats_with_usg(season: str = "2024-25") -> pd.DataFrame:
    """
    使用nba-api获取球员统计数据（包括出勤率、年龄和USG%）
    
    参数:
        season: 赛季，格式如 "2024-25"
    
    返回:
        包含球员统计的DataFrame（包含年龄和USG%）
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        from nba_api.stats.library.parameters import SeasonType
    except ImportError:
        print("错误: 需要安装nba-api")
        print("请运行: pip install nba-api")
        return pd.DataFrame()

    print(f"\n正在获取{season}赛季球员统计数据...")
    
    try:
        # 方法1: 获取基础统计
        print("  [1/2] 获取基础统计...")
        stats_basic = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=SeasonType.regular,
            per_mode_detailed='Totals'
        )
        df_basic = stats_basic.get_data_frames()[0]
        
        time.sleep(1)  # 礼貌性延迟
        
        # 方法2: 获取高级统计（包含USG%）
        print("  [2/2] 获取高级统计（USG%）...")
        stats_advanced = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=SeasonType.regular,
            measure_type_detailed_defense='Advanced'  # ⭐关键：获取高级统计
        )
        df_advanced = stats_advanced.get_data_frames()[0]
        
        # 合并两个数据集
        # 基础统计的关键列（⭐已添加AGE）
        basic_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE',
            'GP', 'MIN',
            'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT',
            'FTM', 'FTA', 'FT_PCT',
            'REB', 'AST', 'STL', 'BLK', 'TOV',
            'PTS', 'PLUS_MINUS'
        ]
        
        # 高级统计的关键列
        advanced_cols = [
            'PLAYER_ID',
            'USG_PCT',      # 使用率
            'TS_PCT',       # 真实命中率
            'EFG_PCT',      # 有效命中率
            'OFF_RATING',   # 进攻效率
            'DEF_RATING',   # 防守效率
            'NET_RATING',   # 净效率
            'AST_PCT',      # 助攻率
            'REB_PCT',      # 篮板率
            'PIE'           # 球员影响力评估
        ]
        
        # 只保留存在的列
        basic_available = [col for col in basic_cols if col in df_basic.columns]
        advanced_available = [col for col in advanced_cols if col in df_advanced.columns]
        
        df_basic_clean = df_basic[basic_available].copy()
        df_advanced_clean = df_advanced[advanced_available].copy()
        
        # 基于PLAYER_ID合并
        df = pd.merge(
            df_basic_clean,
            df_advanced_clean,
            on='PLAYER_ID',
            how='left'
        )
        
        # 计算出勤率相关指标
        if 'GP' in df.columns:
            df['ATTENDANCE_RATE'] = (df['GP'] / 82 * 100).round(2)
            
        if 'MIN' in df.columns and 'GP' in df.columns:
            df['MPG'] = (df['MIN'] / df['GP']).round(1)
        
        print(f"✓ 成功获取统计数据: {df.shape[0]} 名球员, {df.shape[1]} 个字段")
        print(f"✓ 包含年龄: {'AGE' in df.columns}")
        print(f"✓ 包含USG%: {'USG_PCT' in df.columns}")
        
        return df
        
    except Exception as e:
        print(f"获取NBA数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def fetch_player_stats_advanced_only(season: str = "2024-25") -> pd.DataFrame:
    """
    简化版：只获取高级统计（已包含场均数据、年龄和USG%）
    
    这个方法更简单，一次API调用即可
    """
    try:
        from nba_api.stats.endpoints import leaguedashplayerstats
        from nba_api.stats.library.parameters import SeasonType
    except ImportError:
        print("错误: 需要安装nba-api")
        print("请运行: pip install nba-api")
        return pd.DataFrame()

    print(f"\n正在获取{season}赛季球员统计数据（含年龄和USG%）...")
    
    try:
        # 直接获取高级统计（PerGame模式下也包含基础数据）
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=SeasonType.regular,
            measure_type_detailed_defense='Advanced',  # ⭐关键修改
            per_mode_detailed='PerGame'  # 使用场均而非总计
        )
        
        df = stats.get_data_frames()[0]
        
        # 选择关键列
        key_cols = [
            'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'AGE',
            'GP', 'MIN',
            'USG_PCT',      # ⭐使用率
            'TS_PCT',       # 真实命中率
            'EFG_PCT',      # 有效命中率
            'OFF_RATING',   # 进攻效率
            'DEF_RATING',   # 防守效率
            'NET_RATING',   # 净效率
            'AST_PCT',      # 助攻率
            'AST_TO',       # 助攻失误比
            'AST_RATIO',    # 助攻比率
            'OREB_PCT',     # 进攻篮板率
            'DREB_PCT',     # 防守篮板率
            'REB_PCT',      # 篮板率
            'TM_TOV_PCT',   # 失误率
            'PIE',          # 球员影响力评估
            'PACE'          # 节奏
        ]
        
        # 只保留存在的列
        available_cols = [col for col in key_cols if col in df.columns]
        df = df[available_cols].copy()
        
        # 计算出勤率
        if 'GP' in df.columns:
            df['ATTENDANCE_RATE'] = (df['GP'] / 82 * 100).round(2)
        
        print(f"✓ 成功获取统计数据: {df.shape[0]} 名球员, {df.shape[1]} 个字段")
        print(f"✓ 包含年龄: {'AGE' in df.columns}")
        print(f"✓ 包含USG%: {'USG_PCT' in df.columns}")
        
        return df
        
    except Exception as e:
        print(f"获取NBA数据时出错: {e}")
        return pd.DataFrame()


def fetch_team_stats(season: str = "2024-25") -> pd.DataFrame:
    """
    获取球队统计数据
    """
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        from nba_api.stats.library.parameters import SeasonType
    except ImportError:
        return pd.DataFrame()

    print(f"\n正在获取{season}赛季球队统计数据...")
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=SeasonType.regular
        )
        
        df = stats.get_data_frames()[0]
        print(f"✓ 成功获取球队数据: {df.shape[0]} 支球队")
        return df
    except Exception as e:
        print(f"获取球队数据时出错: {e}")
        return pd.DataFrame()


# ============ 第三部分：数据合并 ============

def merge_rapm_and_stats(
    rapm_df: pd.DataFrame, 
    stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    合并RAPM数据和nba-api统计数据
    
    参数:
        rapm_df: RAPM数据
        stats_df: nba-api统计数据
    
    返回:
        合并后的DataFrame
    """
    print("\n正在合并数据...")
    
    # 清理球员姓名以便匹配
    if 'Player' in rapm_df.columns:
        rapm_df['Player_Clean'] = rapm_df['Player'].str.strip()
    
    if 'PLAYER_NAME' in stats_df.columns:
        stats_df['Player_Clean'] = stats_df['PLAYER_NAME'].str.strip()
    
    # 基于球员姓名合并
    merged = pd.merge(
        rapm_df,
        stats_df,
        left_on='Player_Clean',
        right_on='Player_Clean',
        how='left',  # 保留所有RAPM数据
        suffixes=('_RAPM', '_NBA')
    )
    
    # 删除临时列
    if 'Player_Clean' in merged.columns:
        merged = merged.drop(columns=['Player_Clean'])
    
    print(f"✓ 合并完成: {merged.shape[0]} 行, {merged.shape[1]} 列")
    
    matched = merged['PLAYER_NAME'].notna().sum()
    unmatched = len(merged) - matched
    print(f"  匹配成功: {matched} 名球员")
    if unmatched > 0:
        print(f"  未匹配: {unmatched} 名球员")
    
    return merged


# ============ 主函数 ============

def main():
    """主函数：执行完整的数据获取流程"""
    
    print("=" * 60)
    print("NBA球员数据获取工具（改进版 - 包含年龄）")
    print("=" * 60)
    
    # 1. 从本地HTML文件获取RAPM数据
    rapm_df = None
    try:
        rapm_df = fetch_xrapm_table("RAPM_3y.html")
        rapm_df.to_csv("csv_fold/xRAPM.csv", index=False, encoding="utf-8-sig")
        print("✓ RAPM数据已保存: xRAPM.csv")
        
        # 显示数据预览
        print("\nRAPM数据预览（前5行）:")
        preview_cols = ['Player', 'Team', 'Offense_value', 'Defense_value', 'Total_value']
        available_preview = [col for col in preview_cols if col in rapm_df.columns]
        print(rapm_df[available_preview].head().to_string(index=False))
        
    except Exception as e:
        print(f"✗ 获取RAPM数据失败: {e}")
        return  # 如果RAPM数据获取失败，提前退出
    
    time.sleep(1)  # 礼貌性延迟
    
    # 2. 获取NBA统计数据
    stats_df = None
    try:
        stats_df = fetch_player_stats_with_usg(season="2024-25")
        if not stats_df.empty:
            stats_df.to_csv("csv_fold/nba_player_stats.csv", index=False, encoding="utf-8-sig")
            print("\n✓ NBA统计数据已保存: nba_player_stats.csv")
    except Exception as e:
        print(f"✗ 获取NBA统计数据失败: {e}")
        print("  提示: 请确保已安装nba-api (pip install nba-api)")
    
    # 3. 合并数据（如果两者都成功）
    if rapm_df is not None and stats_df is not None and not stats_df.empty:
        try:
            merged_df = merge_rapm_and_stats(rapm_df, stats_df)
            merged_df.to_csv("csv_fold/nba_combined_data.csv", index=False, encoding="utf-8-sig")
            print("✓ 合并数据已保存: nba_combined_data.csv")
            
            # 显示关键列的预览
            print("\n合并数据预览（前5行，关键列）:")
            key_preview_cols = [
                'Player', 'Team', 'AGE',
                'Total_value', 'Offense_value', 'Defense_value',
                'GP', 'ATTENDANCE_RATE', 'MPG', 'PTS'
            ]
            available_key = [col for col in key_preview_cols if col in merged_df.columns]
            print(merged_df[available_key].head().to_string(index=False))
            
        except Exception as e:
            print(f"✗ 合并数据失败: {e}")
    
    print("\n" + "=" * 60)
    print("数据获取完成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  1. xRAPM.csv - RAPM数据（含数值和百分位数）")
    if stats_df is not None and not stats_df.empty:
        print("  2. nba_player_stats.csv - NBA官方统计数据（含年龄和出勤率）")
    if rapm_df is not None and stats_df is not None and not stats_df.empty:
        print("  3. nba_combined_data.csv - 合并数据")
    
    print("\n数据说明:")
    print("  - AGE：球员年龄")
    print("  - _value列：RAPM的实际数值")
    print("  - _percentile列：该数值在所有球员中的百分位排名")
    print("  - ATTENDANCE_RATE：出勤率（%）")
    print("  - MPG：场均上场时间（分钟）")


if __name__ == "__main__":
    main()

