"""
Test: maybe the creator joined on NAME (not pfr_id/cfb_id).
The 27 remaining FP cases all have valid pfr_ids in draft_picks,
but if the JOIN was by name, these would fail if the names differ.

Let's check: for each FP case, does the combine name match the dp name?
"""
import pandas as pd
import numpy as np
import re

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
train = pd.read_csv('input/train.csv')
train['clean_school'] = train['School'].apply(clean_school)
comb_by_season = {s: g.copy() for s, g in comb.groupby('season')}

# Build dp lookup by season+name (various name cleaning)
def clean_name(n):
    if not isinstance(n, str): return ''
    return re.sub(r'[^a-z]', '', n.lower())

def clean_name_nosuffix(n):
    if not isinstance(n, str): return ''
    s = n.lower().strip()
    s = re.sub(r'\b(jr|sr|ii|iii|iv|v)\.?\s*$', '', s).strip()
    return re.sub(r'[^a-z]', '', s)

# Build lookup tables
dp_by_season_name = {}  # (season, clean_name) -> row
dp_by_season_name_nosuffix = {}  # (season, clean_name_nosuffix) -> row
for _, r in dp.iterrows():
    key1 = (r['season'], clean_name(r['pfr_player_name']))
    key2 = (r['season'], clean_name_nosuffix(r['pfr_player_name']))
    dp_by_season_name[key1] = r
    dp_by_season_name_nosuffix[key2] = r

fp_ids = [284, 334, 399, 420, 576, 579, 780, 854, 908, 977, 1252, 1301, 1648, 1757,
          1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685]

print("=== Name-based join analysis for 27 FP cases ===\n")
for tid in fp_ids:
    row = train[train['Id']==tid].iloc[0]
    year, h_m, w_kg, school = int(row['Year']), row['Height'], row['Weight'], row['clean_school']
    
    year_comb = comb_by_season.get(year, pd.DataFrame())
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    school_match = exact[exact['clean_school'] == school]
    cand = school_match.iloc[0] if len(school_match) > 0 else (exact.iloc[0] if len(exact) > 0 else None)
    
    if cand is None:
        continue
    
    c_name = cand['player_name']
    c_name_clean = clean_name(c_name)
    c_name_nosuffix = clean_name_nosuffix(c_name)
    
    # Check name-based join
    key1 = (year, c_name_clean)
    key2 = (year, c_name_nosuffix)
    dp_by_name = dp_by_season_name.get(key1)
    dp_by_nosuffix = dp_by_season_name_nosuffix.get(key2)
    
    hit = (dp_by_name is not None) or (dp_by_nosuffix is not None)
    
    print(f"ID={tid}: combine='{c_name}' clean='{c_name_clean}' nosuffix='{c_name_nosuffix}'")
    if dp_by_name is not None:
        print(f"  EXACT NAME MATCH: dp='{dp_by_name['pfr_player_name']}' season={dp_by_name['season']}")
    elif dp_by_nosuffix is not None:
        print(f"  NOSUFFIX MATCH: dp='{dp_by_nosuffix['pfr_player_name']}' season={dp_by_nosuffix['season']}")
    else:
        # Check what dp names exist for this season
        dp_year = dp[dp['season']==year]
        last = c_name.split()[-1].lower()
        similar = dp_year[dp_year['pfr_player_name'].str.lower().str.contains(last, na=False)]
        if not similar.empty:
            print(f"  NO MATCH, but similar last names: {similar['pfr_player_name'].tolist()}")
        else:
            print(f"  NO MATCH at all in dp for year {year}")
