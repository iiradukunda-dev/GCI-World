"""
Complete final verification of ALL 238 test predictions = 0 (not drafted).
For each, check:
1. Exact measurement match in combine → undrafted (draft_ovr = NaN)
2. Cross-check by combine player name against draft_picks
3. Use pfr_id to verify they're NOT in draft_picks
4. Check combine_pro_day for alternative measurements

Flag any player where the combine match has draft_ovr but we predict 0.
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
test['clean_school'] = test['School'].apply(clean_school)
combine_by_season = {s: g.copy() for s, g in combine.groupby('season')}

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

dp_pfr = dict(zip(draft_picks['pfr_player_id'].dropna(), zip(draft_picks['round'], draft_picks['pick'])))

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

# Verify all 238 predictions=0
anomalies = []
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 0.0: continue
    
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    
    # Our prediction is 0 — verify the match is truly undrafted
    if pd.notnull(best['draft_ovr']):
        # Mismatch! Our match is drafted but we predict 0 — this shouldn't happen
        # unless this is a known override
        anomalies.append({
            'Id': test_id,
            'Year': row['Year'], 'Position': row['Position'], 'School': row['School'],
            'match_name': best['player_name'], 'match_draft_ovr': best['draft_ovr'],
            'match_pfr_id': best['pfr_id'], 'penalty': round(pen, 4),
        })

print(f"Test predictions=0 where match has draft_ovr (anomalies): {len(anomalies)}")
print()
if anomalies:
    for a in anomalies:
        print(f"  ID {a['Id']}: {a['Year']} {a['School']} {a['Position']} -> '{a['match_name']}' (draft_ovr={a['match_draft_ovr']}, pfr_id={a['match_pfr_id']}, pen={a['penalty']:.2f})")
else:
    print("All 0-predictions correctly matched to undrafted combine players!")

print()
# Now check: for all 238 NOT DRAFTED predictions,
# is there any same-school+year DRAFTED player with very close measurements?
# These are "near-miss FNs" from the reverse direction.
print("="*70)
print("Checking: are there drafted combine players very close to our 0-predictions?")
print("(Within 0.5in height AND 5lbs weight, same school)")
print()
near_miss = []
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 0.0: continue
    
    year = row['Year']
    cs = row['clean_school']
    h_m, w_kg = row['Height'], row['Weight']
    
    # Check for drafted players at same school/year within small margin
    same_school = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == cs) &
        (combine['draft_ovr'].notna()) &
        (np.abs(combine['height_m'] - h_m) < 0.015) &  # 0.5in
        (np.abs(combine['weight_kg'] - w_kg) < 2.3)    # 5lbs
    ]
    
    if not same_school.empty:
        for _, dr in same_school.iterrows():
            h_diff_in = abs(dr['height_m'] - h_m) / 0.0254
            w_diff_lb = abs(dr['weight_kg'] - w_kg) / 0.45359237
            
            # Get our actual match
            result = get_best_match(row, combine_by_season)
            our_match = result[1]['player_name'] if result[0] is not None else 'NONE'
            
            near_miss.append({
                'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
                'our_match': our_match,
                'close_drafted': dr['player_name'], 'drafted_ovr': dr['draft_ovr'],
                'h_diff_in': round(h_diff_in, 2), 'w_diff_lb': round(w_diff_lb, 2),
            })

print(f"Found {len(near_miss)} 0-predictions with close drafted combine players:")
if near_miss:
    df_nm = pd.DataFrame(near_miss).sort_values(['h_diff_in', 'w_diff_lb'])
    print(df_nm[['Id','Year','Position','School','our_match','close_drafted','drafted_ovr','h_diff_in','w_diff_lb']].to_string())
