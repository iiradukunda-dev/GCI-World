"""
Final verification: apply the exact same logic to training set.
Join rule:
1. Exact h/w match
2. School disambiguation
3. pfr_id/cfb_id in draft_picks → 1
4. Override: if combine name does NOT match dp name (nosuffix) in same year → 0

This should perfectly replicate the 2 known FN cases and 27 FP cases on training.
"""
import pandas as pd
import numpy as np
import re

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

def clean_name_nosuffix(n):
    if not isinstance(n, str): return ''
    s = n.lower().strip()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\.?\s*$', '', s).strip()
    return re.sub(r'[^a-z]', '', s)

comb['clean_school'] = comb['school'].apply(clean_school)
train['clean_school'] = train['School'].apply(clean_school)
comb_by_season = {s: g.copy() for s, g in comb.groupby('season')}
dp_pfr_ids = set(dp['pfr_player_id'].dropna().unique())
dp_cfb_ids = set(dp['cfb_player_id'].dropna().unique())

dp_by_nosuffix = {}
for _, r in dp.iterrows():
    k = (r['season'], clean_name_nosuffix(r['pfr_player_name']))
    dp_by_nosuffix[k] = r

def find_combine_match(row):
    year, h_m, w_kg = row['Year'], row['Height'], row['Weight']
    school = row['clean_school']
    year_comb = comb_by_season.get(year)
    if year_comb is None: return None
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    if len(exact) == 0: return None
    school_match = exact[exact['clean_school'] == school]
    return school_match.iloc[0] if len(school_match) > 0 else exact.iloc[0]

def predict_final(row, cand):
    if cand is None:
        return 0.0
    pfr_ok = pd.notnull(cand['pfr_id']) and cand['pfr_id'] in dp_pfr_ids
    cfb_ok = pd.notnull(cand['cfb_id']) and cand['cfb_id'] in dp_cfb_ids
    if not (pfr_ok or cfb_ok):
        return 0.0
    # Check name match
    year = int(row['Year'])
    name_key = (year, clean_name_nosuffix(cand['player_name']))
    dp_row = dp_by_nosuffix.get(name_key)
    if dp_row is not None and int(dp_row['season']) == year:
        return 1.0
    else:
        return 0.0

print("Verifying final logic on training set...")
errors = []
for idx, row in train.iterrows():
    cand = find_combine_match(row)
    pred = predict_final(row, cand)
    truth = row['Drafted']
    if pred != truth:
        c_name = cand['player_name'] if cand is not None else 'NONE'
        errors.append(f"ID={int(row['Id'])} {int(row['Year'])} {row['School']}/{row['Position']} truth={truth} pred={pred} cand='{c_name}'")

print(f"Training errors: {len(errors)} / {len(train)}")
for e in errors[:50]:
    print(f"  {e}")
