import pandas as pd
import numpy as np
import re

# Team abbreviation to full name mapping
team_mapping = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards'
}

def clean_rapm_value(value_str):
    """Remove parentheses and percentile from RAPM value, keep only the number"""
    if pd.isna(value_str):
        return np.nan
    # Extract the number before the parentheses
    match = re.match(r'(-?\d+\.?\d*)', str(value_str))
    if match:
        return float(match.group(1))
    return np.nan

def read_rapm_file(year):
    """Read RAPM file for a specific year and clean the data"""
    filepath = f'RAPM_CSV/{year}.csv'
    try:
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {year}.csv with any encoding")
        
        df['year'] = year
        
        # Clean RAPM values - remove parentheses
        for col in ['Offense', 'Defense(*)', 'Total']:
            if col in df.columns:
                df[col] = df[col].apply(clean_rapm_value)
        
        # Rename columns for consistency
        df = df.rename(columns={'Defense(*)': 'Defense'})
        
        return df[['Player', 'year', 'Total']]
    except Exception as e:
        print(f"Error reading {year}.csv: {e}")
        return None

# Step 1: Read all RAPM files
print("Step 1: Reading RAPM files...")
rapm_data = []
for year in range(2015, 2024):
    df = read_rapm_file(year)
    if df is not None:
        rapm_data.append(df)
        print(f"  Loaded {year}: {len(df)} players")

rapm_df = pd.concat(rapm_data, ignore_index=True)
rapm_df = rapm_df.rename(columns={'Total': 'RAPM'})
print(f"Total RAPM records: {len(rapm_df)}")

# Step 2: Read person_time.csv
print("\nStep 2: Reading person_time.csv...")
person_time_df = pd.read_csv('person_time.csv')
print(f"Total person_time records: {len(person_time_df)}")

# Step 3: Merge RAPM with person_time
print("\nStep 3: Merging RAPM with person_time...")
merged_df = pd.merge(
    rapm_df,
    person_time_df,
    on=['Player', 'year'],
    how='inner'
)
print(f"Merged records: {len(merged_df)}")

# Save team.csv: (player, year, team, RAPM, time)
team_df = merged_df[['Player', 'year', 'Team', 'RAPM', 'Minutes']].copy()
team_df = team_df.rename(columns={'Minutes': 'time'})
team_df.to_csv('team.csv', index=False)
print(f"Saved team.csv with {len(team_df)} records")

# Step 4: Calculate team-weighted RAPM
print("\nStep 4: Calculating team-weighted RAPM...")

# Group by team and year to calculate weighted RAPM
team_stats = []

for (team_abbr, year), group in merged_df.groupby(['Team', 'year']):
    total_time = group['Minutes'].sum()
    
    if total_time > 0:
        # Calculate weighted RAPM
        weighted_rapm = (group['RAPM'] * group['Minutes']).sum() / total_time
        
        team_stats.append({
            'Team': team_abbr,
            'year': year,
            'RAPM': weighted_rapm
        })

team_rapm_df = pd.DataFrame(team_stats)

# Convert team abbreviation to full name
team_rapm_df['team'] = team_rapm_df['Team'].map(team_mapping)

# Drop the abbreviation column
team_rapm_df = team_rapm_df[['team', 'year', 'RAPM']]

# Save team_average.csv
team_rapm_df.to_csv('team_average.csv', index=False)
print(f"Saved team_average.csv with {len(team_rapm_df)} records")

# Step 5: Read nba_team_stat.csv and merge with team RAPM
print("\nStep 5: Creating regression.csv...")
nba_stats_df = pd.read_csv('TQ_stat/nba_team_stat.csv')

# Merge team RAPM with NBA stats
regression_df = pd.merge(
    nba_stats_df,
    team_rapm_df,
    on=['team', 'year'],
    how='inner'
)

# Select final columns: (team, year, RAPM, win_pct)
regression_df = regression_df[['team', 'year', 'RAPM', 'win_pct']]

# Save regression.csv
regression_df.to_csv('regression.csv', index=False)
print(f"Saved regression.csv with {len(regression_df)} records")

print("\n=== Summary ===")
print(f"team.csv: {len(team_df)} player-team-year records")
print(f"team_average.csv: {len(team_rapm_df)} team-year RAPM values")
print(f"regression.csv: {len(regression_df)} team-year records with win_pct")
print("\nAll files saved successfully!")
