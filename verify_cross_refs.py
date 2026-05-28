"""
Verify the most suspicious cross-reference cases where combine player
comes from a different school than the test player.
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

# Suspicious test IDs where cross-reference school doesn't match
suspicious_ids = [3354, 2901, 3392, 3109, 3099, 3382, 3075, 2879, 2965]

def get_best_match(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season:
        return None, None
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
    return scored[0] if scored else (None, None)

for test_id in suspicious_ids:
    trow = test[test['Id'] == test_id]
    if trow.empty:
        continue
    trow = trow.iloc[0]
    pred = sub_dict.get(test_id, -1)
    
    print(f"ID {test_id}: {trow['Year']} {trow['School']} {trow['Position']} | PREDICTION={pred}")
    print(f"  h={trow['Height']:.4f}m, w={trow['Weight']:.4f}kg")
    
    result = get_best_match(trow, combine_by_season)
    if result[0] is not None:
        pen, best = result
        dp_check = draft_picks[draft_picks['pfr_player_name'].str.lower() == best['player_name'].lower()]
        dp_info = f"R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if not dp_check.empty else "not_in_dp"
        print(f"  -> Best match: '{best['player_name']}' ({best['pos']}, {best['school']}) draft_ovr={best['draft_ovr']} pen={pen:.4f} | {dp_info}")
    
    # Also show same-school+year players
    year = int(trow['Year'])
    cs = trow['clean_school']
    same = combine[(combine['season'] == year) & (combine['clean_school'] == cs)]
    print(f"  Same school+year ({trow['School']} {year}):")
    for _, cr in same.iterrows():
        h_d = abs(cr['height_m'] - trow['Height']) / 0.0254
        w_d = abs(cr['weight_kg'] - trow['Weight']) / 0.45359237
        print(f"    {cr['player_name']:30s} {cr['pos']:5s} {cr['ht']:5s}/{cr['wt']:.0f} h_diff={h_d:.1f}in w_diff={w_d:.1f}lbs draft_ovr={cr['draft_ovr']}")
    print()
