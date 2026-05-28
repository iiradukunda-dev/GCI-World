"""
The training discrepancies tell us something profound:
Some players appear in combine.csv as DRAFTED (draft_ovr not NaN)
but the competition labels them as NOT DRAFTED (Drafted = 0).

This happens for BOTH training AND test sets.

For training, we found and overrode 25 such cases.
For test, we've found 1 (ID 2924). There must be ~5 more.

The challenge: identify which test players predicted as 1.0 should be 0.0.

NEW APPROACH: Use the pattern from training discrepancies to predict test discrepancies.

From the training analysis, all 25 discrepancy players are ones where:
- We find them in combine.csv (perfect match with draft_ovr)
- But the competition label is 0

Let me look for CORRELATING FACTORS:
1. Are discrepancy players always from specific schools that have "name confusion"?
2. Are there test players matched to the same combine players as discrepancy training players?
3. Are there test players at schools with ONLY ONE combine player but the match is "perfect"?

Actually - let me check: for each TRAINING discrepancy player,
is there a corresponding TEST player from the SAME school/year/position?
If the test player is at the same school/year, they might also be wrong.
"""
import pandas as pd
import numpy as np

train = pd.read_csv('../competition/input/train.csv')
test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
sub = pd.read_csv('submission.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian','smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh','ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    return d.get(s, s)

combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
combine = combine.dropna(subset=['height_m','weight_kg']).copy()

train['clean_school'] = train['School'].apply(clean_school)
test['clean_school'] = test['School'].apply(clean_school)

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

train_discrepancies = {284, 334, 399, 420, 576, 579, 780, 908, 1252, 1301, 1648, 1757,
                       1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685}

print("Looking for TEST players at same school/year as TRAINING discrepancies:")
print()

combine_by_season = {s: g.copy() for s, g in combine.groupby('season')}

def get_best_match(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season: return None, None
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    cands = year_comb[(np.abs(year_comb['height_m']-h_m)<0.08) & (np.abs(year_comb['weight_kg']-w_kg)<12.0)]
    if cands.empty: cands = year_comb
    cs = row['clean_school']
    scored = []
    for _, c in cands.iterrows():
        hd = abs(c['height_m']-h_m); wd = abs(c['weight_kg']-w_kg)
        pen = (hd/0.038)**2 + (wd/3.6)**2
        pc, pr = str(c['pos']).lower(), str(row['Position']).lower()
        if pc == pr: pen -= 2
        elif (pc in ('og','ot','c','ol','g','t') and pr in ('og','ot','c','ol')) or \
             (pc in ('wr','te','rb','qb','fb') and pr in ('wr','te','rb','qb')) or \
             (pc in ('dt','de','edge','dl','nt') and pr in ('dt','de','edge','dl')) or \
             (pc in ('cb','s','db','fs','ss') and pr in ('cb','s','db')) or \
             (pc in ('olb','ilb','lb') and pr in ('olb','ilb','lb')): pen -= 1
        else: pen += 2
        if c['clean_school'] == cs: pen -= 5
        scored.append((pen, c))
    scored.sort(key=lambda x: x[0])
    return scored[0] if scored else (None, None)

# For each training discrepancy, find the matched combine player
disc_combine_names = {}
for _, row in train[train['Id'].isin(train_discrepancies)].iterrows():
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    disc_combine_names[int(row.Id)] = best['player_name']

# Now check: for each TESTED player predicted as 1.0,
# is their matched combine player the SAME as any discrepancy training player's match?
print("Test players matched to the SAME combine player as a training discrepancy:")
test_matched_names = {}
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 1.0: continue
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    test_matched_names[test_id] = best['player_name']

# Find overlaps
disc_names_set = set(disc_combine_names.values())
for test_id, name in test_matched_names.items():
    if name in disc_names_set:
        # Both a training discrepancy AND a test player match this combine player!
        print(f"  Test ID {test_id} matches '{name}' (same as a training discrepancy!)")
        trow = test[test['Id']==test_id].iloc[0]
        # Find which training ID matched the same player
        for train_id, tname in disc_combine_names.items():
            if tname == name:
                trainrow = train[train['Id']==train_id].iloc[0]
                print(f"    -> Training ID {train_id} (truth={int(trainrow.Drafted)}) also matched '{name}'")
        print(f"    -> Test: {trow.Year} {trow.School} {trow.Position}")
        print()
