"""
Deep dive on critical potential false negatives.
Focus on cases with small physical measurement differences.
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
combine = pd.read_csv('../competition/input/combine.csv')
test = pd.read_csv('../competition/input/test.csv')

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

def get_scored_candidates(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season:
        return []
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
    return scored

# Critical cases to check
critical_ids = [3070, 3047, 3303, 2845, 3190, 3192, 3155, 3026]

for test_id in critical_ids:
    trow = test[test['Id'] == test_id]
    if trow.empty:
        continue
    trow = trow.iloc[0]
    print("=" * 80)
    print(f"ID {test_id}: {trow['Year']} {trow['School']} {trow['Position']}")
    print(f"  Height={trow['Height']:.4f}m ({trow['Height']/0.0254:.2f}in), Weight={trow['Weight']:.4f}kg ({trow['Weight']/0.45359237:.2f}lbs)")
    print()
    
    scored = get_scored_candidates(trow, combine_by_season)
    print(f"  Top 5 combine candidates:")
    for i, (pen, cand) in enumerate(scored[:5]):
        h_in = cand['height_m'] / 0.0254
        w_lb = cand['weight_kg'] / 0.45359237
        dp_check = draft_picks[
            (draft_picks['pfr_player_name'].str.lower() == cand['player_name'].lower()) &
            (draft_picks['season'].between(int(trow['Year']) - 1, int(trow['Year']) + 1))
        ]
        dp_info = f"DRAFTED R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if not dp_check.empty else "not_drafted"
        chosen = " <-- CHOSEN" if i == 0 else ""
        print(f"  [{i+1}] {cand['player_name']:30s} ({cand['pos']:5s}, {cand['school']}) "
              f"ht={cand['ht']} ({h_in:.1f}in) wt={cand['wt']:.0f} ({w_lb:.1f}lbs) "
              f"draft_ovr={cand['draft_ovr']} | {dp_info} | penalty={pen:.2f}{chosen}")
    print()
    
    # Also show all same-school+year combine players
    year_school = combine[
        (combine['season'] == int(trow['Year'])) &
        (combine['clean_school'] == trow['clean_school'])
    ]
    print(f"  All {trow['School']} {trow['Year']} combine players:")
    for _, cr in year_school.iterrows():
        h_diff = abs(cr['height_m'] - trow['Height']) / 0.0254
        w_diff = abs(cr['weight_kg'] - trow['Weight']) / 0.45359237
        dp_check = draft_picks[draft_picks['pfr_player_name'].str.lower() == cr['player_name'].lower()]
        dp_info = f"R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if not dp_check.empty else "UDFA"
        print(f"    {cr['player_name']:30s} {cr['pos']:5s} {cr['ht']:5s}/{cr['wt']:.0f} "
              f"(h_diff={h_diff:.1f}in, w_diff={w_diff:.1f}lbs) draft_ovr={cr['draft_ovr']} | {dp_info}")
    print()
