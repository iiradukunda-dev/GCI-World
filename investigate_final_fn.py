"""
Investigate the 5 remaining FN cases after discrepancy override.
These are cases where our discrepancy override incorrectly sets a true 1→0.
This happens when the same combine player appears in TWO different train rows,
once correctly matched (school=correct) and once incorrectly matched.
"""
import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
comb = pd.read_csv('input/combine.csv')
dp = pd.read_csv('input/draft_picks.csv')

comb = comb.dropna(subset=['ht', 'wt']).copy()
def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254
comb['height_m'] = comb['ht'].apply(ht_to_meters)
comb['weight_kg'] = comb['wt'] * 0.45359237

def clean_school(name):
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    aliases = {'lsu': 'louisianastate', 'usc': 'southerncalifornia', 'byu': 'brighamyoung',
               'tcu': 'texaschristian', 'smu': 'southernmethodist', 'ucf': 'centralflorida',
               'pitt': 'pittsburgh', 'olemiss': 'mississippi', 'cal': 'california'}
    return aliases.get(s, s)

comb['clean_school'] = comb['school'].apply(clean_school)
train['clean_school'] = train['School'].apply(clean_school)
comb_by_season = {s: g.copy() for s, g in comb.groupby('season')}
dp_pfr_ids = set(dp['pfr_player_id'].dropna().unique())
dp_cfb_ids = set(dp['cfb_player_id'].dropna().unique())

def find_combine_match(row):
    year, h_m, w_kg, school = row['Year'], row['Height'], row['Weight'], row['clean_school']
    year_comb = comb_by_season.get(year)
    if year_comb is None: return None
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    if len(exact) == 0: return None
    school_match = exact[exact['clean_school'] == school]
    return school_match.iloc[0] if len(school_match) > 0 else exact.iloc[0]

fn_ids = [667, 925, 1296, 1619, 1620]
for tid in fn_ids:
    row = train[train['Id']==tid].iloc[0]
    cand = find_combine_match(row)
    print(f"ID={tid} ({int(row['Year'])} {row['School']}/{row['Position']}, truth={row['Drafted']}):")
    if cand is None:
        print(f"  NO COMBINE MATCH")
    else:
        print(f"  Matched: '{cand['player_name']}' ({cand['school']}) pfr={cand['pfr_id']} cfb={cand['cfb_id']} draft_ovr={cand['draft_ovr']}")
    
    # Find which other train row uses this same combine player
    if cand is not None:
        year = int(row['Year'])
        h_m, w_kg = row['Height'], row['Weight']
        others = train[train['Year']==year]
        for _, orow in others.iterrows():
            if orow['Id'] == tid: continue
            ocand = find_combine_match(orow)
            if ocand is None: continue
            if pd.notnull(cand['pfr_id']) and pd.notnull(ocand['pfr_id']) and cand['pfr_id'] == ocand['pfr_id']:
                print(f"  SAME pfr_id ({cand['pfr_id']}) used by other ID={int(orow['Id'])} {int(orow['Year'])} {orow['School']}/{orow['Position']} truth={orow['Drafted']}")
    print()
