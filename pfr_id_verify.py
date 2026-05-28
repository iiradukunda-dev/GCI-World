"""
Use pfr_id from combine.csv as the definitive identifier.
For each test player predicted as DRAFTED (1.0):
1. Find their matched combine player
2. Get that combine player's pfr_id
3. Look up pfr_id in draft_picks
4. Verify they were truly drafted

Also use cfb_id to cross-check.

The KEY insight: if a combine player has draft_ovr but their pfr_id
has draft_picks round=7 or some specific round, maybe the labels
use a different round cutoff?

Wait - let me check: are the training discrepancies (truth=0 but combine says drafted)
players drafted in a specific round (e.g., all round 7 or later)?
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
combine_by_season = {s: g.copy() for s, g in combine.groupby('season')}

train_discrepancies = {284, 334, 399, 420, 576, 579, 780, 908, 1252, 1301, 1648, 1757,
                       1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685}

dp_by_pfr = dict(zip(draft_picks['pfr_player_id'].dropna(), draft_picks.to_dict('records')))

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

# Analyze the 25 training discrepancies using pfr_id
print("TRAINING DISCREPANCIES - pfr_id analysis:")
print("(These are truth=0 cases where combine says drafted)")
print()
discrepancy_rounds = []
for _, row in train[train['Id'].isin(train_discrepancies)].iterrows():
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    pfr_id = best['pfr_id'] if pd.notna(best['pfr_id']) else None
    dp_info = dp_by_pfr.get(pfr_id, {})
    rnd = dp_info.get('round', None)
    pick = dp_info.get('pick', None)
    discrepancy_rounds.append(rnd)
    print(f"ID {int(row.Id):4d} truth={int(row.Drafted)} | '{best.player_name}' ({best.school}) "
          f"draft_ovr={best.draft_ovr} | round={rnd} pick={pick}")

print()
print(f"Round distribution of training discrepancies: {sorted([r for r in discrepancy_rounds if r])}")
print()

# Now analyze ALL test predictions (1.0) using pfr_id  
print("\nTEST PREDICTIONS = 1.0 - pfr_id analysis:")
print("Sorted by draft round (late rounds are most likely to be errors)")
print()
sub_dict = dict(zip(sub['Id'], sub['Drafted']))

test_results = []
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 1.0: continue
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    pfr_id = best['pfr_id'] if pd.notna(best['pfr_id']) else None
    dp_info = dp_by_pfr.get(pfr_id, {})
    rnd = dp_info.get('round', None)
    pick = dp_info.get('pick', None)
    test_results.append({
        'Id': test_id, 'Year': row['Year'], 'Position': row['Position'], 'School': row['School'],
        'match_name': best['player_name'], 'draft_ovr': best['draft_ovr'],
        'pfr_id': pfr_id, 'round': rnd, 'pick': pick, 'penalty': round(pen, 4),
    })

df_test = pd.DataFrame(test_results)
# Sort by round (None = not in draft_picks by pfr_id, then late rounds)
df_test['round_sort'] = df_test['round'].apply(lambda x: x if x else 999)
df_test = df_test.sort_values('round_sort', ascending=False)

# Show cases where round is None (pfr_id not in draft_picks!)
null_round = df_test[df_test['round'].isna()]
print(f"Test predictions where pfr_id not found in draft_picks: {len(null_round)}")
print(null_round[['Id','Year','Position','School','match_name','draft_ovr','pfr_id','round']].to_string())
print()
print(f"Late rounds (7th) predicted as drafted:")
round7 = df_test[df_test['round'] == 7]
print(round7[['Id','Year','Position','School','match_name','draft_ovr','round','pick']].to_string())
