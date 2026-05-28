"""
Focused analysis of the 17 FN cases.
For each FN case, find:
1. The row our mapping picks (wrong - undrafted)
2. The correct combine row (drafted) for that player's school+year
"""
import pandas as pd
import numpy as np

train = pd.read_csv('../competition/input/train.csv')
comb = pd.read_csv('../competition/input/combine.csv')
dp = pd.read_csv('../competition/input/draft_picks.csv')

comb = comb.dropna(subset=['ht', 'wt']).copy()
def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

comb['height_m'] = comb['ht'].apply(ht_to_meters)
comb['weight_kg'] = comb['wt'] * 0.45359237
comb_by_season = {s: g.copy() for s, g in comb.groupby('season')}
dp_pfr_ids = set(dp['pfr_player_id'].dropna().unique())
dp_cfb_ids = set(dp['cfb_player_id'].dropna().unique())

fn_ids = [139, 301, 313, 452, 462, 492, 523, 696, 895, 1049, 1620, 1798, 2028, 2079, 2332, 2442, 2759]

for tid in fn_ids:
    row = train[train['Id']==tid].iloc[0]
    year, h_m, w_kg = int(row['Year']), row['Height'], row['Weight']
    pos, school = row['Position'], row['School']
    school_clean = school.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','').replace('university','').replace('univ','').replace('state','st')
    
    year_comb = comb_by_season.get(year, pd.DataFrame())
    if year_comb.empty:
        print(f"ID={tid}: No combine data for year {year}")
        continue
    
    # EXACT height/weight match
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    
    print(f"\nID={tid} ({year} {school} {pos}, truth=1.0):")
    print(f"  Height={h_m:.4f}m ({h_m/0.0254:.0f}in), Weight={w_kg:.4f}kg ({w_kg/0.45359237:.0f}lbs)")
    print(f"  Exact h/w matches in combine ({len(exact)}):")
    for _, c in exact.iterrows():
        pfr_m = pd.notnull(c['pfr_id']) and c['pfr_id'] in dp_pfr_ids
        cfb_m = pd.notnull(c['cfb_id']) and c['cfb_id'] in dp_cfb_ids
        c_school = c['school'].lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','').replace('university','').replace('univ','').replace('state','st')
        school_match = (c_school == school_clean)
        print(f"    '{c['player_name']}' ({c['school']}) {c['pos']} pfr={c['pfr_id']} cfb={c['cfb_id']} draft_ovr={c['draft_ovr']} pfr_in_dp={pfr_m} cfb_in_dp={cfb_m} school_match={school_match}")
