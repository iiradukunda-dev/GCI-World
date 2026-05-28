"""
Use combine_pro_day.csv to get player names for ALL test players.
Strategy: match test h/w to combine_pro_day h/w
- Test h is in meters from combine.csv (integer inches converted)
- combine_pro_day has decimal inches
- Match by rounding combine_pro_day height to nearest integer

Key: combine_pro_day should have PLAYER NAMES which can be cross-checked with draft_picks.
For test players predicted as 1.0 whose names we can find in combine_pro_day,
verify they're in draft_picks.

Also: combine_pro_day has both combine AND pro day measurements.
Need to check if the integer values match.
"""
import pandas as pd
import numpy as np

test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
combine_pro = pd.read_csv('../competition/input/combine_pro_day.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
sub = pd.read_csv('submission.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian',
         'smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh',
         'ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    return d.get(s, s)

combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
combine = combine.dropna(subset=['height_m','weight_kg']).copy()
test['clean_school'] = test['School'].apply(clean_school)
sub_dict = dict(zip(sub['Id'], sub['Drafted']))

# Round combine_pro heights to integer inches for matching
combine_pro['height_in_int'] = combine_pro['Height (in)'].round(0).astype('Int64')
combine_pro['weight_lbs_int'] = combine_pro['Weight (lbs)'].round(0).astype('Int64')
combine_pro['height_m'] = combine_pro['Height (in)'] * 0.0254
combine_pro['weight_kg'] = combine_pro['Weight (lbs)'] * 0.45359237
combine_pro['clean_school'] = combine_pro['College'].apply(clean_school)

dp_by_name = {}
for _, row in draft_picks.iterrows():
    dp_by_name[row['pfr_player_name'].lower()] = row

# Match test players to combine_pro_day
print("Matching test players to combine_pro_day (by year + integer height + weight):")
print()
errors = []
for _, row in test.iterrows():
    test_id = row['Id']
    pred = sub_dict.get(test_id, 0)
    
    year = row['Year']
    h_m = row['Height']
    w_kg = row['Weight']
    h_in = round(h_m / 0.0254)  # integer inches
    w_lb = round(w_kg / 0.45359237)  # integer lbs
    cs = row['clean_school']
    
    # Match in combine_pro_day by year + integer height + weight + school
    exact = combine_pro[
        (combine_pro['Year'] == year) &
        (combine_pro['height_in_int'] == h_in) &
        (combine_pro['weight_lbs_int'] == w_lb) &
        (combine_pro['clean_school'] == cs)
    ]
    
    if len(exact) == 1:
        player_name = exact.iloc[0]['player']
        
        # Check if this player is in draft_picks
        dp_row = dp_by_name.get(player_name.lower())
        in_dp = dp_row is not None
        if not in_dp:
            # Try name variations
            for key in dp_by_name:
                if key.split()[-1] == player_name.lower().split()[-1] and abs(dp_by_name[key]['season'] - year) <= 1:
                    dp_row = dp_by_name[key]
                    in_dp = True
                    break
        
        dp_status = f"R{dp_row['round']}P{dp_row['pick']}" if in_dp else "UDFA"
        
        # Check for mismatch between our prediction and draft status
        if pred == 1.0 and not in_dp:
            errors.append({'Id': test_id, 'Year': year, 'Position': row['Position'],
                          'School': row['School'], 'player_name': player_name,
                          'pred': pred, 'dp_status': dp_status, 'type': 'FP'})
        elif pred == 0.0 and in_dp:
            errors.append({'Id': test_id, 'Year': year, 'Position': row['Position'],
                          'School': row['School'], 'player_name': player_name,
                          'pred': pred, 'dp_status': dp_status, 'type': 'FN'})

print(f"Errors found (mismatches between combine_pro_day name and draft_picks): {len(errors)}")
for e in errors:
    print(f"  ID {e['Id']} {e['Year']} {e['School']} {e['Position']}: '{e['player_name']}' pred={e['pred']} dp={e['dp_status']} type={e['type']}")
