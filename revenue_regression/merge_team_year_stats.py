# merge_team_year_stats.py
# 生成合并后的 CSV： (Team, year, win_pct, win_pct(last_year), Player Expenses(M$), Revenue(M$))
# year 范围：2016-2024（含）

import pandas as pd

TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


def merge_stats(
    nba_csv: str,
    revenue_csv: str,
    out_csv: str,
    start_year: int = 2015,
    end_year: int = 2024,
) -> pd.DataFrame:
    # 1) 读取
    nba = pd.read_csv(nba_csv)
    rev = pd.read_csv(revenue_csv)

    # 2) 清洗/标准化列
    if "team" not in nba.columns or "year" not in nba.columns or "win_pct" not in nba.columns:
        raise ValueError("nba_csv 必须至少包含列：team, year, win_pct")

    nba["team"] = nba["team"].astype(str).str.strip()
    nba["year"] = nba["year"].astype(int)

    if "Team" not in rev.columns or "year" not in rev.columns:
        raise ValueError("revenue_csv 必须至少包含列：Team, year")
    if "Player Expenses(M$)" not in rev.columns or "Revenue(M$)" not in rev.columns:
        raise ValueError("revenue_csv 必须包含列：Player Expenses(M$), Revenue(M$)")

    rev["Team"] = rev["Team"].astype(str).str.strip().str.upper()
    rev["year"] = rev["year"].astype(int)

    # 3) team 全称 -> 缩写，便于和 revenue 表对齐
    nba["Team"] = nba["team"].map(TEAM_NAME_TO_ABBR)
    missing = nba.loc[nba["Team"].isna(), "team"].unique()
    if len(missing) > 0:
        raise ValueError(f"以下球队名称无法映射到缩写，请补充映射表：{missing.tolist()}")

    # 4) 计算 win_pct(last_year)
    nba = nba.sort_values(["Team", "year"])
    nba["win_pct(last_year)"] = nba.groupby("Team")["win_pct"].shift(1)

    # 5) 过滤年份（注意：win_pct(last_year) 需要上一年数据，因此原表最好包含 start_year-1）
    nba = nba[(nba["year"] >= start_year) & (nba["year"] <= end_year)].copy()
    rev = rev[(rev["year"] >= start_year) & (rev["year"] <= end_year)].copy()

    # 6) 合并
    rev_keep = rev[["Team", "year", "Player Expenses(M$)", "Revenue(M$)"]].copy()
    merged = (
        nba[["Team", "year", "win_pct", "win_pct(last_year)"]]
        .merge(rev_keep, on=["Team", "year"], how="inner", validate="one_to_one")
        .sort_values(["Team", "year"])
    )

    # 7) 输出指定列顺序
    merged = merged[
        ["Team", "year", "win_pct", "win_pct(last_year)", "Player Expenses(M$)", "Revenue(M$)"]
    ]
    merged.to_csv(out_csv, index=False)
    return merged


if __name__ == "__main__":
    # 按需改成你的路径
    nba_csv_path = "TQ_stat/nba_team_stat.csv"
    revenue_csv_path = "csv_fold/revenue_stat.csv"
    out_csv_path = "revenue_regression/merged_team_year_stats_2016_2024.csv"

    df = merge_stats(nba_csv_path, revenue_csv_path, out_csv_path, 2016, 2024)
    print("Saved:", out_csv_path)
    print("Rows:", len(df))
    print(df.head(10).to_string(index=False))
