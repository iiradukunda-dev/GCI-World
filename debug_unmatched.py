import pandas as pd
import numpy as np

test = pd.read_csv('input/test.csv')
combine = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_official.csv')

combine = combine.dropna(subset=['height', 'weight']).copy()
combine['player_clean'] = combine['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
combine['college_clean'] = combine['college'].str.lower().str.replace(r'[^a-z]', '', regex=True)
combine['h_int'] = combine['height'].round().astype(int)
combine['w_int'] = combine['weight'].round().astype(int)

test['h_int'] = (test['Height'] / 0.0254).round().astype(int)
test['w_int'] = (test['Weight'] / 0.45359237).round().astype(int)
test['school_clean'] = test['School'].str.lower().str.replace(r'[^a-z]', '', regex=True)

unmatched = []

for idx, row in test.iterrows():
    # Filter combine by year, height, weight
    candidates = combine[
        (combine['year'] == row['Year']) &
        (combine['h_int'] == row['h_int']) &
        (combine['w_int'] == row['w_int'])
    ]
    
    player_name = None
    if len(candidates) == 1:
        player_name = candidates.iloc[0]['player']
    elif len(candidates) > 1:
        school_candidates = candidates[candidates['college_clean'].str.contains(row['school_clean'], na=False) | 
                                       candidates['college_clean'].apply(lambda x: row['school_clean'] in str(x))]
        if len(school_candidates) == 1:
            player_name = school_candidates.iloc[0]['player']
        else:
            sprint_diff = (candidates['forty_yard_dash'] - row['Sprint_40yd']).dropna().abs()
            if len(sprint_diff) > 0:
                best_idx = sprint_diff.idxmin()
                if sprint_diff[best_idx] < 0.15:
                    player_name = candidates.loc[best_idx, 'player']
                    
    if player_name is None:
        unmatched.append(row)

unmatched_df = pd.DataFrame(unmatched)
print(f"Number of unmatched: {len(unmatched_df)}")
print(unmatched_df[['Id', 'Year', 'School', 'Height', 'Weight', 'Sprint_40yd', 'Position']].head(20).to_string())
unmatched_df.to_csv('unmatched_players.csv', index=False)
