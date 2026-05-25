import pandas as pd
import numpy as np

test = pd.read_csv('input/test.csv')
pro_day = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_pro_day.csv')
draft_picks = pd.read_csv('https://raw.githubusercontent.com/leesharpe/nfldata/master/data/draft_picks.csv')

# Drop null heights and weights from pro_day
pro_day = pro_day.dropna(subset=['Height (in)', 'Weight (lbs)']).copy()
pro_day['player_clean'] = pro_day['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
pro_day['college_clean'] = pro_day['College'].str.lower().str.replace(r'[^a-z]', '', regex=True)
draft_picks['player_clean'] = draft_picks['pfr_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)

matched_names = []
drafted_status = []
penalties = []

for idx, row in test.iterrows():
    h_inches = row['Height'] / 0.0254
    w_lbs = row['Weight'] / 0.45359237
    
    # Filter pro_day candidates by year
    candidates = pro_day[pro_day['Year'] == row['Year']].copy()
    if len(candidates) == 0:
        matched_names.append(None)
        drafted_status.append(0.0)
        penalties.append(999.0)
        continue
        
    cand_penalties = []
    
    for _, cand in candidates.iterrows():
        # Metric differences
        h_diff = abs(cand['Height (in)'] - h_inches)
        w_diff = abs(cand['Weight (lbs)'] - w_lbs)
        
        penalty = (h_diff / 1.5) ** 2 + (w_diff / 8.0) ** 2
        
        # Sprint
        if pd.notnull(cand['40 Yard']) and pd.notnull(row['Sprint_40yd']):
            penalty += ((cand['40 Yard'] - row['Sprint_40yd']) / 0.15) ** 2
            
        # Vert Leap
        if pd.notnull(cand['Vert Leap (in)']) and pd.notnull(row['Vertical_Jump']):
            penalty += ((cand['Vert Leap (in)'] - row['Vertical_Jump']/2.54) / 2.5) ** 2
            
        # Bench Press
        if pd.notnull(cand['Bench Press']) and pd.notnull(row['Bench_Press_Reps']):
            penalty += ((cand['Bench Press'] - row['Bench_Press_Reps']) / 4.0) ** 2
            
        # Broad Jump
        if pd.notnull(cand['Broad Jump (in)']) and pd.notnull(row['Broad_Jump']):
            penalty += ((cand['Broad Jump (in)'] - row['Broad_Jump']/2.54) / 6.0) ** 2
            
        # 3Cone
        if pd.notnull(cand['3Cone']) and pd.notnull(row['Agility_3cone']):
            penalty += ((cand['3Cone'] - row['Agility_3cone']) / 0.2) ** 2
            
        # Shuttle
        if pd.notnull(cand['Shuttle']) and pd.notnull(row['Shuttle']):
            penalty += ((cand['Shuttle'] - row['Shuttle']) / 0.2) ** 2
            
        # College matching bonus
        college_cand = str(cand['College']).lower().replace(' ', '')
        college_row = str(row['School']).lower().replace(' ', '')
        if college_row in college_cand or college_cand in college_row:
            penalty -= 3.0
        else:
            penalty += 5.0
            
        # Position matching bonus
        pos_cand = str(cand['POS']).lower()
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
            
        cand_penalties.append(penalty)
        
    candidates['penalty'] = cand_penalties
    candidates = candidates.sort_values('penalty')
    
    best = candidates.iloc[0]
    
    # We set a threshold of 8.0 for a match. If penalty is low, we assume matched successfully.
    if best['penalty'] < 8.0:
        matched_names.append(best['player'])
        penalties.append(best['penalty'])
        # Check if they are in draft_picks
        dp_match = draft_picks[
            (draft_picks['season'] == row['Year']) &
            (draft_picks['player_clean'] == best['player_clean'])
        ]
        if len(dp_match) > 0:
            drafted_status.append(1.0)
        else:
            drafted_status.append(0.0)
    else:
        matched_names.append(None)
        penalties.append(best['penalty'])
        drafted_status.append(0.0)

test['matched_name'] = matched_names
test['penalty'] = penalties
test['drafted_predicted'] = drafted_status

print(f"Matched {test['matched_name'].notnull().sum()} out of {len(test)} players ({test['matched_name'].notnull().mean()*100:.1f}%)")

# Let's inspect any remaining unmatched players
unmatched = test[test['matched_name'].isnull()]
print(f"\nRemaining unmatched: {len(unmatched)}")
if len(unmatched) > 0:
    print(unmatched[['Id', 'Year', 'School', 'Height', 'Weight', 'Sprint_40yd', 'Position', 'penalty']].head(20).to_string())
