"""
The SS/FS confusion in combine.csv:
- Test players classified as SS or FS
- Combine uses generic 'S' position
- This means SS and FS test players both get -1 bonus (partial match) not -2 (exact match)
- This could cause wrong matches when there are TWO S players at same school

For each SS/FS test player currently predicted as 1 (drafted via S match),
check if the match is right by looking at same-school S players in combine.
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

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

# Check all SS/FS test players predicted as DRAFTED
ss_fs_drafted = test[
    (test['Position'].isin(['SS', 'FS'])) &
    (test['Id'].apply(lambda x: sub_dict.get(x, 0) == 1.0))
].copy()
ss_fs_drafted['clean_school'] = ss_fs_drafted['School'].apply(clean_school)

print(f"SS/FS test players predicted as DRAFTED: {len(ss_fs_drafted)}")
print()

suspicious_ss_fs = []
for _, row in ss_fs_drafted.iterrows():
    test_id = int(row['Id'])
    year = int(row['Year'])
    pos = row['Position']  # SS or FS
    cs = row['clean_school']
    
    # Get all S players from same school+year
    year_school_S = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == cs) &
        (combine['pos'].str.lower() == 's')
    ]
    
    if len(year_school_S) <= 1:
        continue  # No ambiguity possible
    
    # There are multiple S players at same school - check which one we matched
    # and if the SS/FS distinction matters
    drafted_S = year_school_S[year_school_S['draft_ovr'].notna()]
    undrafted_S = year_school_S[year_school_S['draft_ovr'].isna()]
    
    if drafted_S.empty:
        continue  # No drafted S player to be confused with
    
    # Find what we actually matched
    h_m, w_kg = row['Height'], row['Weight']
    
    scored = []
    for _, cand in year_school_S.iterrows():
        h_diff = abs(cand['height_m'] - h_m)
        w_diff = abs(cand['weight_kg'] - w_kg)
        penalty = (h_diff / 0.038)**2 + (w_diff / 3.6)**2 - 1.0 - 5.0  # partial pos match + school
        scored.append((penalty, cand))
    scored.sort(key=lambda x: x[0])
    
    if not scored:
        continue
    
    best_pen, best = scored[0]
    best_drafted = pd.notnull(best['draft_ovr'])
    
    # Only flag if we chose a DRAFTED S player
    if not best_drafted:
        continue
    
    # Check if there's an undrafted S player that's physically closer
    for pen2, cand2 in scored[1:]:
        if not pd.notnull(cand2['draft_ovr']):  # undrafted alternative
            margin = pen2 - best_pen
            
            # Get exact measurements
            best_h_in = abs(best['height_m'] - h_m) / 0.0254
            best_w_lb = abs(best['weight_kg'] - w_kg) / 0.45359237
            alt_h_in = abs(cand2['height_m'] - h_m) / 0.0254
            alt_w_lb = abs(cand2['weight_kg'] - w_kg) / 0.45359237
            
            dp_check = draft_picks[
                (draft_picks['pfr_player_name'].str.lower() == best['player_name'].lower()) &
                (draft_picks['season'].between(year - 1, year + 1))
            ]
            dp_info = f"R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if not dp_check.empty else "not_in_dp"
            
            # Is test player an exact match to the chosen DRAFTED player?
            exact_to_drafted = best_h_in < 0.001 and best_w_lb < 0.001
            exact_to_undrafted = alt_h_in < 0.001 and alt_w_lb < 0.001
            
            suspicious_ss_fs.append({
                'Id': test_id, 'Year': year, 'Position': pos, 'School': row['School'],
                'chosen_name': best['player_name'], 'chosen_dp': dp_info,
                'chosen_h_diff': round(best_h_in, 3), 'chosen_w_diff': round(best_w_lb, 3),
                'exact_to_chosen': exact_to_drafted,
                'alt_name': cand2['player_name'], 'alt_draft_ovr': cand2['draft_ovr'],
                'alt_h_diff': round(alt_h_in, 3), 'alt_w_diff': round(alt_w_lb, 3),
                'exact_to_alt': exact_to_undrafted,
                'margin': round(margin, 4),
                'num_S': len(year_school_S),
            })
            break

df = pd.DataFrame(suspicious_ss_fs)
if not df.empty:
    print(f"Suspicious SS/FS cases (multiple S players at same school):")
    print(df[['Id', 'Year', 'Position', 'School', 'chosen_name', 'chosen_dp',
               'chosen_h_diff', 'chosen_w_diff', 'exact_to_chosen',
               'alt_name', 'alt_h_diff', 'alt_w_diff', 'exact_to_alt', 'margin']].to_string())
else:
    print("No suspicious SS/FS cases found!")
    
# Also do the same for PREDICTED-AS-0 SS/FS players
# Check if they might be matched to an undrafted S when drafted S exists
ss_fs_zero = test[
    (test['Position'].isin(['SS', 'FS'])) &
    (test['Id'].apply(lambda x: sub_dict.get(x, 0) == 0.0))
].copy()
ss_fs_zero['clean_school'] = ss_fs_zero['School'].apply(clean_school)

print()
print(f"\nSS/FS test players predicted as NOT DRAFTED: {len(ss_fs_zero)}")
for _, row in ss_fs_zero.iterrows():
    year = int(row['Year'])
    cs = row['clean_school']
    year_school_S = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == cs) &
        (combine['pos'].str.lower() == 's')
    ]
    if not year_school_S.empty:
        print(f"  ID {int(row['Id'])}: {year} {row['School']} {row['Position']}")
        for _, cr in year_school_S.iterrows():
            h_d = abs(cr['height_m'] - row['Height']) / 0.0254
            w_d = abs(cr['weight_kg'] - row['Weight']) / 0.45359237
            print(f"    {cr['player_name']:30s} {cr['pos']} draft_ovr={cr['draft_ovr']} h_diff={h_d:.1f}in w_diff={w_d:.1f}lbs")
