#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract NYK players from player_scores.csv and output nyk_players.csv
"""

import os
import pandas as pd


def extract_nyk_players(input_csv: str, output_csv: str) -> pd.DataFrame:
    # Read data
    df = pd.read_csv(input_csv)

    # Filter NYK
    nyk_df = df[df["Team"] == "NYK"].copy()

    # Sort by score descending
    nyk_df = nyk_df.sort_values("Athletic_Score", ascending=False)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Save
    nyk_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    # Print summary
    print(f"NYK球队共有 {len(nyk_df)} 名球员")
    print(f"数据已保存至: {output_csv}")
    print("\nNYK球员列表：")
    cols = [c for c in ["Rank", "Player", "Athletic_Score"] if c in nyk_df.columns]
    if cols:
        print(nyk_df[cols].to_string(index=False))
    else:
        print(nyk_df.head().to_string(index=False))

    return nyk_df


def main():
    print("=" * 60)
    print("NYK球队球员数据提取")
    print("=" * 60)

    input_csv = "csv_fold/player_scores.csv"
    output_csv = "csv_fold/nyk_players.csv"  # 你也可以改成 "nyk_players.csv"

    extract_nyk_players(input_csv, output_csv)

    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print("生成文件:")
    print(f"  1. CSV文件: {output_csv}")


if __name__ == "__main__":
    main()
