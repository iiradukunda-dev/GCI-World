"""
Deep investigate the 17 False Negatives: truth=1 but pred=0.
The mapping is picking the WRONG combine row for these players.
We need to find the CORRECT combine row.
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
    
    year_comb = comb_by_season.get(year, pd.DataFrame())
    if year_comb.empty:
        print(f"ID={tid}: No combine data for year {year}")
        continue
    
    # Find all close matches
    cands = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 0.06) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 10.0)
    ]
    
    print(f"\nID={tid} ({year} {school} {pos}, truth=1.0):")
    print(f"  Height={h_m:.4f}m ({h_m/0.0254:.1f}in), Weight={w_kg:.4f}kg ({w_kg/0.45359237:.1f}lbs)")
    print(f"  Close combine matches ({len(cands)}):")
    for _, c in cands.iterrows():
        pfr_match = pd.notnull(c['pfr_id']) and c['pfr_id'] in dp_pfr_ids
        cfb_match = pd.notnull(c['cfb_id']) and c['cfb_id'] in dp_cfb_ids
        drafted = pd.notnull(c['draft_ovr'])
        print(f"    '{c['player_name']}' ({c['school']}) {c['pos']} pfr={c['pfr_id']} cfb={c['cfb_id']} draft={drafted} pfr_in_dp={pfr_match} cfb_in_dp={cfb_match}")
