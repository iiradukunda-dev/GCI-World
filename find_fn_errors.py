"""
Look for False Negatives: test players we predict as 0 (not drafted)
but who might actually be 1 (drafted).

These would be players where:
1. Their correct combine match DOES have draft_ovr (they were drafted)
2. But our algorithm matched them to a DIFFERENT (undrafted) combine player

Also check: are there players in our 238 predicted-as-0 list who appear
in draft_picks.csv by searching school + year + position?
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

# Get our predicted-as-0 players
predicted_zero = test[test['Id'].apply(lambda x: sub_dict.get(x, 0) == 0.0)].copy()
print(f"Players predicted as 0 (not drafted): {len(predicted_zero)}")

# For each, look at ALL combine players at same school+year+position
# and check if any of them in draft_picks
potential_fn = []
for _, row in predicted_zero.iterrows():
    test_id = int(row['Id'])
    year = int(row['Year'])
    pos = row['Position']
    school = row['School']
    cs = row['clean_school']
    
    # Get all same-school+year players from combine
    year_comb = combine_by_season.get(year, pd.DataFrame())
    if year_comb.empty:
        continue
    
    same_school = year_comb[year_comb['clean_school'] == cs]
    
    # Among those, find any that are drafted AND match this position group
    pos_lower = pos.lower()
    pos_group_match = lambda p: (
        (p in ('og', 'ot', 'c', 'ol', 'g', 't') and pos_lower in ('og', 'ot', 'c', 'ol')) or
        (p in ('wr', 'te', 'rb', 'qb', 'fb') and pos_lower in ('wr', 'te', 'rb', 'qb')) or
        (p in ('dt', 'de', 'edge', 'dl', 'nt') and pos_lower in ('dt', 'de', 'edge', 'dl')) or
        (p in ('cb', 's', 'db', 'fs', 'ss') and pos_lower in ('cb', 's', 'db', 'fs', 'ss')) or
        (p in ('olb', 'ilb', 'lb') and pos_lower in ('olb', 'ilb', 'lb')) or
        p == pos_lower
    )
    
    drafted_same_school_pos = same_school[
        same_school.apply(lambda r: pd.notnull(r['draft_ovr']) and pos_group_match(str(r['pos']).lower()), axis=1)
    ]
    
    if not drafted_same_school_pos.empty:
        for _, cr in drafted_same_school_pos.iterrows():
            h_diff = abs(cr['height_m'] - row['Height'])
            w_diff = abs(cr['weight_kg'] - row['Weight'])
            if h_diff < 0.08 and w_diff < 12:  # Close measurements
                potential_fn.append({
                    'Id': test_id, 'Year': year, 'Position': pos, 'School': school,
                    'test_h': row['Height'], 'test_w': row['Weight'],
                    'cand_name': cr['player_name'], 'cand_pos': cr['pos'],
                    'cand_h': cr['height_m'], 'cand_w': cr['weight_kg'],
                    'h_diff_in': round(h_diff / 0.0254, 2),
                    'w_diff_lbs': round(w_diff / 0.45359237, 2),
                    'draft_ovr': cr['draft_ovr'],
                })

df_fn = pd.DataFrame(potential_fn)
print(f"\nPotential False Negatives (predicted 0 but a drafted combine player is nearby):")
print(f"Total: {len(df_fn)}")
print()
if len(df_fn) > 0:
    # Sort by measurement difference
    df_fn['total_diff'] = df_fn['h_diff_in'].abs() + df_fn['w_diff_lbs'].abs() / 10
    df_fn = df_fn.sort_values('total_diff')
    print(df_fn[['Id', 'Year', 'Position', 'School', 'cand_name', 'cand_pos', 'h_diff_in', 'w_diff_lbs', 'draft_ovr']].to_string())
