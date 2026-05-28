"""
Find ALL cases where our algorithm chose the WRONG player due to near-tie scores.
Cases where:
1. We chose an UNDRAFTED player (draft_ovr=NaN) as best match
2. The 2nd best candidate IS drafted
3. The margin between best and 2nd is very small (< 2.0)

These are potential False Negatives that we're missing.
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
combine = pd.read_csv('../competition/input/combine.csv')
test = pd.read_csv('../competition/input/test.csv')
sub = pd.read_csv('submission.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str:
        return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    school_canonical = {
        'lsu': 'louisianastate', 'usc': 'southerncalifornia',
        'byu': 'brighamyoung', 'tcu': 'texaschristian',
        'smu': 'southernmethodist', 'ucf': 'centralflorida',
        'pitt': 'pittsburgh', 'ole miss': 'mississippi', 'olemiss': 'mississippi',
        'cal': 'california'
    }
    if not isinstance(name, str):
        return ''
    s = name.lower().replace(' ', '').replace('.', '').replace('&', '').replace('-', '').replace('(', '').replace(')', '')
    s = s.replace('university', '').replace('univ', '').replace('state', 'st')
    return school_canonical.get(s, s)

combine = combine.dropna(subset=['ht', 'wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
test['clean_school'] = test['School'].apply(clean_school)
combine_by_season = {s: group.copy() for s, group in combine.groupby('season')}

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

# Find all test players currently predicted as 0 (not drafted)
# where a drafted candidate exists with close penalty
fn_candidates = []

for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 0.0:
        continue  # Skip already-drafted predictions
    
    year = row['Year']
    if year not in combine_by_season:
        continue
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    
    cands = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 0.08) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 12.0)
    ]
    if len(cands) == 0:
        cands = year_comb
    
    school_row_clean = row['clean_school']
    scored = []
    for _, cand in cands.iterrows():
        h_diff = abs(cand['height_m'] - h_m)
        w_diff = abs(cand['weight_kg'] - w_kg)
        penalty = (h_diff / 0.038)**2 + (w_diff / 3.6)**2
        pos_cand = str(cand['pos']).lower()
        pos_row = str(row['Position']).lower()
        if pos_cand == pos_row:
            penalty -= 2.0
        elif (pos_cand in ('og', 'ot', 'c', 'ol', 'g', 't') and pos_row in ('og', 'ot', 'c', 'ol')) or \
             (pos_cand in ('wr', 'te', 'rb', 'qb', 'fb') and pos_row in ('wr', 'te', 'rb', 'qb')) or \
             (pos_cand in ('dt', 'de', 'edge', 'dl', 'nt') and pos_row in ('dt', 'de', 'edge', 'dl')) or \
             (pos_cand in ('cb', 's', 'db', 'fs', 'ss') and pos_row in ('cb', 's', 'db', 'fs', 'ss')) or \
             (pos_cand in ('olb', 'ilb', 'lb') and pos_row in ('olb', 'ilb', 'lb')):
            penalty -= 1.0
        else:
            penalty += 2.0
        if cand['clean_school'] == school_row_clean:
            penalty -= 5.0
        scored.append((penalty, cand))
    
    scored.sort(key=lambda x: x[0])
    if not scored:
        continue
    
    best_pen, best = scored[0]
    
    # We chose an undrafted player as best
    if pd.notnull(best['draft_ovr']):
        continue  # Best is drafted - not our case (that's the FP scenario)
    
    # Look for a close drafted alternative
    for i, (pen, cand) in enumerate(scored[1:], 1):
        if pd.notnull(cand['draft_ovr']):
            margin = pen - best_pen
            if margin < 3.0:  # Within 3 penalty units
                # Verify in draft_picks
                dp_check = draft_picks[
                    (draft_picks['pfr_player_name'].str.lower() == cand['player_name'].lower()) &
                    (draft_picks['season'].between(int(year) - 1, int(year) + 1))
                ]
                dp_info = f"R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if not dp_check.empty else "not_in_dp"
                
                fn_candidates.append({
                    'Id': test_id,
                    'Year': year,
                    'Position': row['Position'],
                    'School': row['School'],
                    'chosen_name': best['player_name'],
                    'chosen_pos': best['pos'],
                    'chosen_penalty': round(best_pen, 4),
                    'alt_name': cand['player_name'],
                    'alt_pos': cand['pos'],
                    'alt_school': cand['school'],
                    'alt_penalty': round(pen, 4),
                    'alt_draft_ovr': cand['draft_ovr'],
                    'alt_in_dp': dp_info,
                    'margin': round(margin, 4),
                    'rank': i,
                })
                break  # Only take first drafted alternative

df_fn = pd.DataFrame(fn_candidates)
print(f"Potential FN cases (predicted 0, but close drafted alternative exists): {len(df_fn)}")
print()
if not df_fn.empty:
    df_fn = df_fn.sort_values('margin')
    print("Sorted by margin (closest first):")
    pd.set_option('display.max_colwidth', 30)
    print(df_fn[['Id', 'Year', 'Position', 'School', 'chosen_name', 
                  'alt_name', 'alt_pos', 'alt_draft_ovr', 'alt_in_dp', 'margin']].to_string())
    df_fn.to_csv('fn_candidates.csv', index=False)
    print('\nSaved fn_candidates.csv')
