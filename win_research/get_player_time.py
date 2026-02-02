"""
Script to fetch NBA player minutes played data
Using nba_api to get player time for seasons 2015-2024
"""

from nba_api.stats.endpoints import leaguedashplayerstats
from nba_api.stats.static import teams
import pandas as pd
import time

def get_season_string(year):
    """Convert year to season string, e.g. 2015 -> '2014-15'"""
    return f"{year-1}-{str(year)[2:]}"

def get_player_time_for_season(year):
    """Get player stats for a specific season"""
    season = get_season_string(year)
    print(f"Fetching data for {season} season...")
    
    try:
        # Get player stats for this season
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star='Regular Season'
        )
        
        df = stats.get_data_frames()[0]
        
        # Select needed columns
        # MIN = total minutes played
        # TEAM_ABBREVIATION = team abbreviation
        # PLAYER_NAME = player name
        df_selected = df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'MIN']].copy()
        df_selected['year'] = year
        
        # Rename columns
        df_selected.columns = ['Player', 'Team', 'Minutes', 'year']
        
        return df_selected
    
    except Exception as e:
        print(f"Error fetching {season} season data: {e}")
        return None

def main():
    all_data = []
    
    # Fetch data for seasons 2015-2024
    for year in range(2015, 2025):
        df = get_player_time_for_season(year)
        if df is not None:
            all_data.append(df)
            print(f"Successfully fetched {year} season data, {len(df)} records")
        
        # Avoid rate limiting
        time.sleep(1)
    
    # Combine all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        output_file = 'person_time.csv'
        final_df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
        print(f"Total records: {len(final_df)}")
        print(f"\nFirst few rows:")
        print(final_df.head(10))
    else:
        print("Failed to fetch any data")

if __name__ == "__main__":
    main()
