"""
Find all test-year cases where two combine players from same school/year/position
have identical or very similar measurements.

These are the cases where our matching algorithm might pick the wrong player
(predicting 1 when truth is 0, or vice versa).

This IS the training discrepancy mechanism:
- Training discrepancy: Train player's measurements EXACTLY match a DRAFTED combine player
- But the train player is ACTUALLY an UNDRAFTED player from the same school
- They just happen to have the same/similar measurements

For test: same pattern possible.
- If a test player's measurements exactly match a DRAFTED combine player
- But there's also an UNDRAFTED player at same school with same/similar measurements
- We might be predicting 1 when truth is 0

Key: look for cases where the TOP match is DRAFTED but there's a CLOSE second that is UNDRAFTED
(and they're from the same school, so the school bonus doesn't differentiate them)
"""
import pandas as pd
import numpy as np

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

def get_scored(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season: return []
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
    return scored

# Find test cases predicted as 1 where the second-best match from same school is UNDRAFTED
# These are the high-risk false positives (same as training discrepancy pattern)
risky_fp = []
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 1.0:
        continue  # Only check predicted-as-drafted players
    
    scored = get_scored(row, combine_by_season)
    if len(scored) < 2: continue
    
    best_pen, best = scored[0]
    if not pd.notnull(best['draft_ovr']): continue  # Best is not drafted, skip
    
    # Look for SAME SCHOOL second-best that is UNDRAFTED
    cs = row['clean_school']
    for pen2, c2 in scored[1:]:
        if c2['clean_school'] != cs: continue  # Must be same school
        margin = pen2 - best_pen
        if margin > 5.0: break  # Too far, not risky
        
        # Found same-school undrafted alternative within 5 units
        risky_fp.append({
            'Id': test_id,
            'Year': row['Year'],
            'Position': row['Position'],
            'School': row['School'],
            'best_name': best['player_name'],
            'best_draft_ovr': best['draft_ovr'],
            'best_pos': best['pos'],
            'best_pen': round(best_pen, 4),
            'best_h_diff': round(abs(best['height_m']-row['Height'])/0.0254, 3),
            'best_w_diff': round(abs(best['weight_kg']-row['Weight'])/0.45359237, 3),
            'alt_name': c2['player_name'],
            'alt_draft_ovr': c2['draft_ovr'],
            'alt_pos': c2['pos'],
            'alt_pen': round(pen2, 4),
            'margin': round(margin, 4),
        })
        break

df = pd.DataFrame(risky_fp)
print(f"Test predictions=1 with same-school UNDRAFTED second-best within 5 units: {len(df)}")
print()
if not df.empty:
    df = df.sort_values('margin')
    print(df[['Id','Year','Position','School','best_name','best_draft_ovr',
               'best_h_diff','best_w_diff','alt_name','alt_draft_ovr','margin']].to_string())

df.to_csv('risky_fp_candidates.csv', index=False)
print("\nSaved risky_fp_candidates.csv")
