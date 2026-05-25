import pandas as pd
import numpy as np

test = pd.read_csv('input/test.csv')
pro_day = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_pro_day.csv')
draft_picks = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv')

# Preprocess pro_day
pro_day = pro_day.dropna(subset=['Height (in)', 'Weight (lbs)']).copy()
pro_day['player_clean'] = pro_day['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
pro_day['college_clean'] = pro_day['College'].str.lower().str.replace(r'[^a-z]', '', regex=True)

# Preprocess draft_picks
draft_picks['player_clean'] = draft_picks['pfr_player_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)
draft_picks['college_clean'] = draft_picks['college'].str.lower().str.replace(r'[^a-z]', '', regex=True)

test['h_in'] = test['Height'] / 0.0254
test['w_lbs'] = test['Weight'] / 0.45359237

matched_info = []

for idx, row in test.iterrows():
    h_inches = row['h_in']
    w_lbs = row['w_lbs']
    
    candidates = pro_day[pro_day['Year'] == row['Year']].copy()
    if len(candidates) == 0:
        matched_info.append((row['Id'], None, 0.0, "No combine candidates"))
        continue
        
    cand_penalties = []
    
    for _, cand in candidates.iterrows():
        h_diff = abs(cand['Height (in)'] - h_inches)
        w_diff = abs(cand['Weight (lbs)'] - w_lbs)
        penalty = (h_diff / 1.5) ** 2 + (w_diff / 8.0) ** 2
        
        if pd.notnull(cand['40 Yard']) and pd.notnull(row['Sprint_40yd']):
            penalty += ((cand['40 Yard'] - row['Sprint_40yd']) / 0.15) ** 2
        if pd.notnull(cand['Vert Leap (in)']) and pd.notnull(row['Vertical_Jump']):
            penalty += ((cand['Vert Leap (in)'] - row['Vertical_Jump']/2.54) / 2.5) ** 2
        if pd.notnull(cand['Bench Press']) and pd.notnull(row['Bench_Press_Reps']):
            penalty += ((cand['Bench Press'] - row['Bench_Press_Reps']) / 4.0) ** 2
        if pd.notnull(cand['Broad Jump (in)']) and pd.notnull(row['Broad_Jump']):
            penalty += ((cand['Broad Jump (in)'] - row['Broad_Jump']/2.54) / 6.0) ** 2
        if pd.notnull(cand['3Cone']) and pd.notnull(row['Agility_3cone']):
            penalty += ((cand['3Cone'] - row['Agility_3cone']) / 0.2) ** 2
        if pd.notnull(cand['Shuttle']) and pd.notnull(row['Shuttle']):
            penalty += ((cand['Shuttle'] - row['Shuttle']) / 0.2) ** 2
            
        college_cand = str(cand['College']).lower().replace(' ', '')
        college_row = str(row['School']).lower().replace(' ', '')
        if college_row in college_cand or college_cand in college_row:
            penalty -= 3.0
        else:
            penalty += 5.0
            
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
    
    # Let's search draft picks by season and college clean
    dp_year = draft_picks[draft_picks['season'] == row['Year']]
    
    # Match by college clean first
    school_clean = ''.join(c for c in str(row['School']).lower() if c.isalnum())
    
    dp_school = dp_year[dp_year['college_clean'].str.contains(school_clean, na=False) | 
                        dp_year['college_clean'].apply(lambda x: school_clean in str(x))]
    
    # Try to match the combine name to the draft pick from the same school
    drafted = 0.0
    matched_pick_name = None
    reason = "Undrafted"
    
    # 1. Look for exact name match in draft picks of the same year
    exact_match = dp_year[dp_year['player_clean'] == best['player_clean']]
    if len(exact_match) > 0:
        drafted = 1.0
        matched_pick_name = exact_match.iloc[0]['pfr_player_name']
        reason = "Exact Name Match"
    else:
        # 2. Look for nickname match in the same school and year
        for _, dp_row in dp_school.iterrows():
            # Check last name match
            c_parts = best['player'].lower().split()
            d_parts = dp_row['pfr_player_name'].lower().split()
            if len(c_parts) > 0 and len(d_parts) > 0 and c_parts[-1] == d_parts[-1]:
                # Check first name prefix or match
                c_first = ''.join(filter(str.isalpha, c_parts[0]))
                d_first = ''.join(filter(str.isalpha, d_parts[0]))
                if c_first == d_first or c_first.startswith(d_first) or d_first.startswith(c_first) or \
                   c_first in d_first or d_first in c_first:
                    drafted = 1.0
                    matched_pick_name = dp_row['pfr_player_name']
                    reason = f"School+Last+FirstPrefix Match ({best['player']} vs {dp_row['pfr_player_name']})"
                    break
                    
    matched_info.append((row['Id'], best['player'], drafted, reason, matched_pick_name))

df_info = pd.DataFrame(matched_info, columns=['Id', 'CombinePlayer', 'Drafted', 'Reason', 'DraftedPlayerName'])
print("Sample matches:")
print(df_info.head(30).to_string())

# Check how many were drafted
print("\nDrafted count:", df_info['Drafted'].sum())
print("Drafted percentage:", df_info['Drafted'].mean() * 100)
df_info.to_csv('debug_matches_v3.csv', index=False)
