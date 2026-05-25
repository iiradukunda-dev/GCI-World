import pandas as pd
import numpy as np

test = pd.read_csv('input/test.csv')
combine = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_official.csv')
draft_picks = pd.read_csv('https://raw.githubusercontent.com/leesharpe/nfldata/master/data/draft_picks.csv')

# Preprocess combine
combine = combine.dropna(subset=['height', 'weight']).copy()
combine['player_clean'] = combine['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
combine['college_clean'] = combine['college'].str.lower().str.replace(r'[^a-z]', '', regex=True)
draft_picks['player_clean'] = draft_picks['pfr_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)

# Load unmatched to see them
unmatched_df = pd.read_csv('unmatched_players.csv')

# Let's write a matching algorithm using a distance penalty
def match_row(row, combine_df):
    h_inches = row['Height'] / 0.0254
    w_lbs = row['Weight'] / 0.45359237
    
    # Filter candidates of the same year
    candidates = combine_df[combine_df['year'] == row['Year']].copy()
    if len(candidates) == 0:
        return None, "No year match"
        
    penalties = []
    player_names = []
    colleges = []
    
    for idx, cand in candidates.iterrows():
        # Compute difference penalties
        h_diff = abs(cand['height'] - h_inches)
        w_diff = abs(cand['weight'] - w_lbs)
        
        # We start with basic height/weight penalty
        penalty = (h_diff / 1.5) ** 2 + (w_diff / 8.0) ** 2
        
        # Physical stats comparisons (if available in both)
        if pd.notnull(cand['forty_yard_dash']) and pd.notnull(row['Sprint_40yd']):
            penalty += ((cand['forty_yard_dash'] - row['Sprint_40yd']) / 0.15) ** 2
            
        if pd.notnull(cand['vertical_jump']) and pd.notnull(row['Vertical_Jump']):
            # Vertical jump in row is in cm, cand is in inches
            v_row_in = row['Vertical_Jump'] / 2.54
            penalty += ((cand['vertical_jump'] - v_row_in) / 2.5) ** 2
            
        if pd.notnull(cand['bench_press']) and pd.notnull(row['Bench_Press_Reps']):
            penalty += ((cand['bench_press'] - row['Bench_Press_Reps']) / 4.0) ** 2
            
        if pd.notnull(cand['broad_jump']) and pd.notnull(row['Broad_Jump']):
            # Broad jump in row is in cm, cand is in inches
            b_row_in = row['Broad_Jump'] / 2.54
            penalty += ((cand['broad_jump'] - b_row_in) / 6.0) ** 2
            
        if pd.notnull(cand['three_cone_drill']) and pd.notnull(row['Agility_3cone']):
            penalty += ((cand['three_cone_drill'] - row['Agility_3cone']) / 0.2) ** 2
            
        if pd.notnull(cand['twenty_yard_shuttle']) and pd.notnull(row['Shuttle']):
            penalty += ((cand['twenty_yard_shuttle'] - row['Shuttle']) / 0.2) ** 2
            
        # School matching bonus/penalty
        school_cand = str(cand['college']).lower().replace(' ', '')
        school_row = str(row['School']).lower().replace(' ', '')
        if school_row in school_cand or school_cand in school_row:
            # Huge bonus for matching school
            penalty -= 3.0
        else:
            # Penalty for completely different school
            penalty += 5.0
            
        # Position matching bonus/penalty
        pos_cand = str(cand['position']).lower()
        pos_row = str(row['Position']).lower()
        if pos_cand == pos_row:
            penalty -= 1.0
        elif (pos_cand in ('og', 'ot', 'c', 'ol') and pos_row in ('og', 'ot', 'c', 'ol')) or \
             (pos_cand in ('wr', 'te', 'rb', 'qb') and pos_row in ('wr', 'te', 'rb', 'qb')) or \
             (pos_cand in ('dt', 'de', 'edge', 'dl') and pos_row in ('dt', 'de', 'edge', 'dl')) or \
             (pos_cand in ('cb', 's', 'db') and pos_row in ('cb', 's', 'db')) or \
             (pos_cand in ('olb', 'ilb', 'lb') and pos_row in ('olb', 'ilb', 'lb')):
            penalty -= 0.5
        else:
            penalty += 2.0
            
        penalties.append(penalty)
        player_names.append(cand['player'])
        colleges.append(cand['college'])
        
    candidates['penalty'] = penalties
    candidates = candidates.sort_values('penalty')
    
    best = candidates.iloc[0]
    # If the penalty is too high (meaning no good match was found at all), return None
    if best['penalty'] > 12.0:
        return None, f"Closest candidate {best['player']} had too high penalty {best['penalty']:.2f}"
        
    return best, f"Matched {best['player']} with penalty {best['penalty']:.2f}"

# Let's test on the unmatched players
print("Testing matching on the 41 previously unmatched players:")
matched_count = 0
for idx, row in unmatched_df.iterrows():
    best_cand, msg = match_row(row, combine)
    if best_cand is not None:
        matched_count += 1
        print(f"Row ID {row['Id']} ({row['School']} {row['Year']}): {msg} (College: {best_cand['college']})")
    else:
        print(f"Row ID {row['Id']} ({row['School']} {row['Year']}): FAIL - {msg}")
        
print(f"\nMatched {matched_count} out of 41 previously unmatched players!")
