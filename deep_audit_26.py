"""
Deep audit of the 26 remaining errors when using pfr_id + cfb_id join.
Goal: understand what exactly causes these 26 mismatches and replicate
the creator's exact label-generation logic.

The best join rule so far: pfr=True, cfb=True, name=none → 26 errors
"""
import pandas as pd
import numpy as np
import re

train = pd.read_csv('../competition/input/train.csv')
comb = pd.read_csv('../competition/input/combine.csv')
dp = pd.read_csv('../competition/input/draft_picks.csv')

comb = comb.dropna(subset=['ht', 'wt']).copy()

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str:
        return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

comb['height_m'] = comb['ht'].apply(ht_to_meters)
comb['weight_kg'] = comb['wt'] * 0.45359237

def clean_school(name):
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian',
         'smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh',
         'ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')')
    s = s.replace('university','').replace('univ','').replace('state','st')
    return d.get(s, s)

comb['clean_school'] = comb['school'].apply(lambda x: x.lower() if isinstance(x, str) else '')
train['clean_school'] = train['School'].apply(lambda x: x.lower() if isinstance(x, str) else '')

comb_by_season = {s: group.copy() for s, group in comb.groupby('season')}

# Map train rows to combine candidates (exact h/w match)
train_to_comb = []
for idx, row in train.iterrows():
    year = row['Year']
    year_comb = comb_by_season.get(year, pd.DataFrame())
    if year_comb.empty:
        train_to_comb.append(None)
        continue
    h_m, w_kg = row['Height'], row['Weight']
    cands = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    if len(cands) == 1:
        train_to_comb.append(cands.iloc[0])
    elif len(cands) > 1:
        # Disambiguate by school
        s_clean = row['School'].lower().strip() if isinstance(row['School'], str) else ''
        match_school = cands[cands['school'].str.lower() == s_clean]
        if len(match_school) == 1:
            train_to_comb.append(match_school.iloc[0])
        else:
            train_to_comb.append(cands.iloc[0])
    else:
        # closest match
        best = year_comb.iloc[((year_comb['height_m'] - h_m)**2 + (year_comb['weight_kg'] - w_kg)**2).values.argmin()]
        train_to_comb.append(best)

dp_pfr_ids = set(dp['pfr_player_id'].dropna().unique())
dp_cfb_ids = set(dp['cfb_player_id'].dropna().unique())

print("=== 26 errors under pfr+cfb ID join ===\n")
errors = []
for idx, row in train.iterrows():
    cand = train_to_comb[idx]
    if cand is None:
        continue
    pfr_match = pd.notnull(cand['pfr_id']) and cand['pfr_id'] in dp_pfr_ids
    cfb_match = pd.notnull(cand['cfb_id']) and cand['cfb_id'] in dp_cfb_ids
    pred = 1.0 if (pfr_match or cfb_match) else 0.0
    if pred != row['Drafted']:
        errors.append({
            'Id': int(row['Id']),
            'Year': int(row['Year']),
            'School': row['School'],
            'Pos': row['Position'],
            'Truth': row['Drafted'],
            'Pred': pred,
            'cand_name': cand['player_name'],
            'cand_pfr': cand['pfr_id'],
            'cand_cfb': cand['cfb_id'],
            'pfr_match': pfr_match,
            'cfb_match': cfb_match,
            'draft_ovr': cand['draft_ovr'],
        })

print(f"Total errors: {len(errors)}")
print()
# Separate FP (pred=1 truth=0) and FN (pred=0 truth=1)
fp = [e for e in errors if e['Pred'] == 1.0 and e['Truth'] == 0.0]
fn = [e for e in errors if e['Pred'] == 0.0 and e['Truth'] == 1.0]
print(f"False Positives (pred=1, truth=0): {len(fp)}")
for e in fp:
    print(f"  ID={e['Id']:4d} {e['Year']} {e['School']}/{e['Pos']} | cand='{e['cand_name']}' pfr={e['cand_pfr']} cfb={e['cand_cfb']} | draft_ovr={e['draft_ovr']}")
print()
print(f"False Negatives (pred=0, truth=1): {len(fn)}")
for e in fn:
    # Look up in dp what pfr_id should be
    dp_match = dp[dp['pfr_player_id'] == e['cand_pfr']] if pd.notnull(e['cand_pfr']) else pd.DataFrame()
    print(f"  ID={e['Id']:4d} {e['Year']} {e['School']}/{e['Pos']} | cand='{e['cand_name']}' pfr={e['cand_pfr']} cfb={e['cand_cfb']} | draft_ovr={e['draft_ovr']}")
    if not dp_match.empty:
        print(f"         pfr found in dp: {dp_match[['pfr_player_name','pfr_player_id','season']].to_string(index=False)}")
