"""
Final comprehensive audit: for EVERY test player predicted as DRAFTED (1.0),
show:
1. The matched combine player's name
2. Whether that name appears in draft_picks
3. The combine draft_ovr value
4. The exact match quality

Focus on finding any case where the match is suspicious and could be wrong.
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

# Current overrides
test_discrepancies = {2924}

dp_names_lower = set(draft_picks['pfr_player_name'].str.lower().unique())

results = []
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 1.0:
        continue
    if test_id in test_discrepancies:
        continue
    
    year = row['Year']
    if year not in combine_by_season:
        continue
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    school_row_clean = row['clean_school']
    
    cands = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 0.08) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 12.0)
    ]
    if len(cands) == 0:
        cands = year_comb
    
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
    name = best['player_name']
    
    # Check if name is in draft_picks (exact or variations)
    in_dp_exact = name.lower() in dp_names_lower
    
    # Try variations
    name_clean = name.lower()
    import re
    name_clean_v = re.sub(r'\s+(jr\.?|sr\.?|ii|iii|iv|v)\s*$', '', name_clean)
    name_clean_v = re.sub(r'[^a-z ]', '', name_clean_v).strip()
    in_dp_clean = name_clean_v in dp_names_lower
    
    # Check by last name in same year
    parts = name.split()
    last_name = parts[-1].lower() if parts else ''
    dp_year_last = draft_picks[
        (draft_picks['pfr_player_name'].str.lower().str.endswith(last_name)) &
        (draft_picks['season'].between(int(year)-1, int(year)+1))
    ]
    
    # Exact measurement match
    h_diff_in = abs(best['height_m'] - h_m) / 0.0254
    w_diff_lb = abs(best['weight_kg'] - w_kg) / 0.45359237
    exact = h_diff_in < 0.001 and w_diff_lb < 0.001
    
    results.append({
        'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
        'match_name': name, 'match_school': best['school'], 'match_pos': best['pos'],
        'match_draft_ovr': best['draft_ovr'],
        'penalty': round(best_pen, 4),
        'h_diff_in': round(h_diff_in, 3),
        'w_diff_lb': round(w_diff_lb, 3),
        'exact_match': exact,
        'in_dp_exact': in_dp_exact,
        'in_dp_clean': in_dp_clean,
        'similar_last_names_dp': len(dp_year_last),
        'school_matches': best['clean_school'] == school_row_clean,
    })

df = pd.DataFrame(results)
print(f"Total predicted as DRAFTED: {len(df)}")
print()

# Find all cases NOT in draft_picks by exact name
not_exact = df[~df['in_dp_exact']].copy()
print(f"Cases where matched combine player NOT in draft_picks by exact name: {len(not_exact)}")
print()
# Among these, how many are in dp by cleaned name?
in_clean = not_exact[not_exact['in_dp_clean']]
print(f"  -> Found by cleaned name: {len(in_clean)}")
still_not_found = not_exact[~not_exact['in_dp_clean']]
print(f"  -> Still not found: {len(still_not_found)}")
print()

# Show not-found cases (these might be training-style discrepancies)
if not still_not_found.empty:
    print("NOT FOUND IN DRAFT_PICKS (potential false positives):")
    print(still_not_found[['Id', 'Year', 'Position', 'School', 'match_name', 
                             'match_school', 'match_pos', 'match_draft_ovr',
                             'penalty', 'h_diff_in', 'w_diff_lb', 'exact_match',
                             'similar_last_names_dp']].to_string())
else:
    print("ALL matched combine players are in draft_picks!")

# Save full report
df.to_csv('full_drafted_audit.csv', index=False)
print("\nSaved full_drafted_audit.csv")
