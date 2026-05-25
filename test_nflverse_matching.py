import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
draft_picks = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv')

# Preprocess college names
draft_picks['college_clean'] = draft_picks['college'].str.lower().str.replace(r'[^a-z]', '', regex=True)
train['school_clean'] = train['School'].str.lower().str.replace(r'[^a-z]', '', regex=True)

# Test matching on train.csv drafted players (Drafted == 1.0)
drafted_train = train[train['Drafted'] == 1.0]

matched_count = 0
total_drafted = len(drafted_train)

for idx, row in drafted_train.iterrows():
    # Filter draft picks by season/year
    dp_year = draft_picks[draft_picks['season'] == row['Year']]
    
    # Try to match by college and position
    matches = dp_year[
        (dp_year['college_clean'] == row['school_clean']) &
        (dp_year['position'].str.lower() == row['Position'].lower())
    ]
    
    if len(matches) > 0:
        matched_count += 1
    else:
        # Fallback: check if the college name partially matches or if position group matches
        # Let's print the unmatched cases to see what is happening
        if matched_count < 10:
            print(f"Unmatched drafted player: Year={row['Year']} School={row['School']} Pos={row['Position']}")
            # print candidates from same school and year
            candidates = dp_year[dp_year['college_clean'].str.contains(row['school_clean'], na=False) | 
                                 dp_year['college_clean'].apply(lambda x: row['school_clean'] in str(x))]
            print("Candidates from same school:")
            print(candidates[['pfr_player_name', 'college', 'position']].to_string(index=False))
            print('-'*50)

print(f"\nMatched {matched_count} out of {total_drafted} drafted players ({matched_count/total_drafted*100:.1f}%)")
