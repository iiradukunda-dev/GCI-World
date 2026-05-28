"""
Investigate the 27 remaining FP cases (pred=1, truth=0).
These are cases where pfr_id/cfb_id of the combine player IS in draft_picks,
but the competition label is 0.

Key question: WHY does the join fail for these?
Hypotheses:
A) The combine's pfr_id matches a DIFFERENT player in draft_picks (wrong season?)
B) There's a season mismatch in the join
C) The combine's pfr_id matches but the college doesn't match in a joined check

Let's check the actual draft_picks row for each combine player's pfr_id.
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

dp_by_pfr = {}
for _, r in dp.iterrows():
    if pd.notnull(r['pfr_player_id']):
        dp_by_pfr[r['pfr_player_id']] = r

fp_ids = [284, 334, 399, 420, 576, 579, 780, 854, 908, 977, 1252, 1301, 1648, 1757,
          1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685]

print("=== 27 FP Cases: Why does the join fail? ===\n")
for tid in fp_ids:
    row = train[train['Id']==tid].iloc[0]
    year, h_m, w_kg, school = int(row['Year']), row['Height'], row['Weight'], row['clean_school']
    
    year_comb = comb_by_season.get(year, pd.DataFrame())
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    # Get the matched candidate (school priority)
    school_match = exact[exact['clean_school'] == school]
    cand = school_match.iloc[0] if len(school_match) > 0 else (exact.iloc[0] if len(exact) > 0 else None)
    
    if cand is None:
        print(f"ID={tid}: No match found")
        continue
    
    pfr_id = cand['pfr_id']
    dp_row = dp_by_pfr.get(pfr_id, None)
    
    print(f"ID={tid} ({year} {row['School']}/{row['Position']}, truth=0):")
    print(f"  Matched: '{cand['player_name']}' ({cand['school']}) draft_ovr={cand['draft_ovr']}")
    print(f"  pfr_id={pfr_id}")
    if dp_row is not None:
        print(f"  In draft_picks: '{dp_row['pfr_player_name']}' season={dp_row['season']} college='{dp_row['college']}'")
        # Check if combine season == dp season
        if int(cand['season']) != int(dp_row['season']):
            print(f"  *** SEASON MISMATCH: combine={cand['season']} dp={dp_row['season']} ***")
        else:
            print(f"  Seasons match: {cand['season']} == {dp_row['season']}")
        # Check if college matches school
        dp_college_clean = clean_school(dp_row['college'])
        comb_school_clean = cand['clean_school']
        if dp_college_clean != comb_school_clean:
            print(f"  College mismatch: combine_school='{cand['school']}' ({comb_school_clean}) dp_college='{dp_row['college']}' ({dp_college_clean})")
        else:
            print(f"  Schools match: {comb_school_clean}")
    else:
        print(f"  NOT FOUND in draft_picks by pfr_id!")
    print()
