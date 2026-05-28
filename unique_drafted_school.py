"""
Final check: For the training discrepancies, the test player matched to a 
UNIQUE drafted combine player (no undrafted alternative at same school).
The truth was 0 because the player is actually NOT that combine player.

For test set: which of our 1-predictions match a UNIQUE drafted combine player
at their school/year with NO undrafted alternative?

These would be candidates for the same type of false positive.
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
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian',
         'smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh',
         'ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
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

# For training discrepancies, how many combine players at same school?
print("Training discrepancy - how many combine players at same school/year?")
for _, tr in train[train['Id'].isin(train_discrepancies)].iterrows():
    same = combine[(combine['season']==tr.Year) & (combine['clean_school']==tr.clean_school)]
    undrafted = same[same['draft_ovr'].isna()]
    drafted = same[same['draft_ovr'].notna()]
    print(f"  ID {int(tr.Id):4d} ({tr.Year} {tr.School}): {len(same)} total, {len(drafted)} drafted, {len(undrafted)} undrafted")

print()
print("="*70)
print("Test predictions=1 where match school has ONLY drafted combine players (no undrafted)")
print("(These are like training discrepancies with no undrafted alternative)")
print()

# For each test player predicted as 1, check if their school/year has undrafted combine players
unique_drafted = []
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

# Cross-check: for ALL test predictions=1, count same-school undrafted players
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 1.0: continue
    
    year = row['Year']
    cs = row['clean_school']
    
    same = combine[(combine['season']==year) & (combine['clean_school']==cs)]
    undrafted = same[same['draft_ovr'].isna()]
    
    if len(undrafted) == 0:
        # School has ONLY drafted combine players - this is a risky case
        # (similar to training discrepancy where school has a unique drafted player)
        result = get_best_match(row, combine_by_season)
        if result[0] is None: continue
        pen, best = result
        unique_drafted.append({
            'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
            'total_at_school': len(same), 'drafted_at_school': len(same),
            'match_name': best['player_name'], 'match_draft_ovr': best['draft_ovr'],
            'penalty': round(pen, 4),
        })

print(f"Test predictions=1 at schools with ONLY drafted combine players: {len(unique_drafted)}")
for ud in sorted(unique_drafted, key=lambda x: x['total_at_school']):
    print(f"  ID {ud['Id']}: {ud['Year']} {ud['School']} {ud['Position']} -> '{ud['match_name']}' (pen={ud['penalty']:.2f}) total={ud['total_at_school']}")
