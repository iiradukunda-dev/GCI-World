"""
Investigate 3 combine players with draft_ovr but pfr_id = NaN.
These are our test predictions 3137, 3158, 3215 matched to:
- Charles Scott (LSU RB 2010)
- Chad Jones (LSU SS 2010)
- Kyle Calloway (Iowa OT 2010)

Also: deeply examine ALL training discrepancies to find the pattern
that distinguishes them from non-discrepancies.
The key question: what is DIFFERENT about these 25 training players
that makes them labeled 0 despite matching a drafted combine player?
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

# =====================================================================
# CHECK 1: The 3 combine players with draft_ovr but pfr_id=NaN
# =====================================================================
null_pfr_drafted = combine[combine['draft_ovr'].notna() & combine['pfr_id'].isna()]
print(f"Combine players with draft_ovr but pfr_id=NaN: {len(null_pfr_drafted)}")
test_year_null = null_pfr_drafted[null_pfr_drafted['season'].between(2009, 2019)]
print(f"In test years (2009-2019): {len(test_year_null)}")
print(test_year_null[['season','player_name','school','pos','ht','wt','draft_ovr','pfr_id']].to_string())
print()

# Verify these by searching draft_picks by name
print("Verifying by name in draft_picks:")
for _, cr in test_year_null.iterrows():
    last = cr['player_name'].split()[-1].lower()
    first = cr['player_name'].split()[0].lower()
    dp = draft_picks[
        (draft_picks['pfr_player_name'].str.lower().str.contains(last, na=False)) &
        (draft_picks['season'].between(int(cr['season'])-1, int(cr['season'])+1))
    ]
    print(f"  '{cr['player_name']}' ({cr['school']}, {cr['pos']}, {cr['season']}, draft_ovr={cr['draft_ovr']}):")
    if not dp.empty:
        print(f"    FOUND: {dp[['pfr_player_name','college','round','pick']].to_string(index=False)}")
    else:
        print(f"    NOT FOUND in draft_picks by last name '{last}'")
    print()

# =====================================================================
# CHECK 2: The training discrepancies - are these players in train.csv
# because they share measurements with a DIFFERENT player (not the drafted one)?
# Let's look at the ACTUAL combine row that matches the training player.
# =====================================================================
print("\n" + "="*80)
print("TRAINING DISCREPANCY DEEP DIVE")
print("Key question: The training player's ACTUAL identity vs the wrong combine match")
print()

train_discrepancies = {284, 334, 399, 420, 576, 579, 780, 908, 1252, 1301, 1648, 1757,
                       1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685}

combine_by_season = {s: g.copy() for s, g in combine.groupby('season')}

def get_scored(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season: return []
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    cands = year_comb[(np.abs(year_comb['height_m']-h_m)<0.08) & (np.abs(year_comb['weight_kg']-w_kg)<12.0)]
    if cands.empty: cands = year_comb
    cs = clean_school(row['School'])
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

# For each training discrepancy:
# 1. Find the top match (which we override)
# 2. Find what the SECOND-best match is (that's the 'true' combine player for this train player)
for _, row in train[train['Id'].isin(train_discrepancies)].iterrows():
    scored = get_scored(row, combine_by_season)
    if not scored: continue
    best_pen, best = scored[0]
    second_pen = scored[1][0] if len(scored) > 1 else None
    second = scored[1][1] if len(scored) > 1 else None
    
    # Is the second-best match the 'true' identity?
    # If the training player is labeled 0, their true combine entry should have draft_ovr=NaN
    second_draft = pd.notnull(second['draft_ovr']) if second is not None else None
    
    print(f"ID {int(row.Id):4d} ({row.Year} {row.School} {row.Position}, truth={int(row.Drafted)}):")
    print(f"  Best:   '{best.player_name}' ({best.school}) draft={bool(pd.notnull(best.draft_ovr))} pfr_id={best.pfr_id} pen={best_pen:.2f}")
    if second is not None:
        print(f"  Second: '{second.player_name}' ({second.school}) draft={second_draft} pen={second_pen:.2f} margin={second_pen-best_pen:.2f}")
    print()
