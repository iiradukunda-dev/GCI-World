"""
FUNDAMENTAL RETHINK: Test if draft_picks.csv is the true label source,
not just combine.csv draft_ovr.

The 25 training discrepancies suggest:
- combine.csv says player was drafted (draft_ovr not NaN)
- But train.csv label is 0 (not drafted)

This might be because:
1. The training labels come from a DIFFERENT source
2. Or: draft_picks.csv is the ACTUAL label source, with combine.csv as features

Strategy:
- For each training player, check if their name appears in draft_picks.csv
- See if draft_picks presence correlates with train.csv label better than combine draft_ovr
"""
import pandas as pd
import numpy as np

train = pd.read_csv('../competition/input/train.csv')
combine = pd.read_csv('../competition/input/combine.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str:
        return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian','smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh','ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    return d.get(s, s)

combine = combine.dropna(subset=['ht','wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
train['clean_school'] = train['School'].apply(clean_school)
combine_by_season = {s: group.copy() for s, group in combine.groupby('season')}

train_discrepancies = {284, 334, 399, 420, 576, 579, 780, 908, 1252, 1301, 1648, 1757,
                       1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685}

def get_best_match(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season:
        return None, None
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    cands = year_comb[(np.abs(year_comb['height_m']-h_m)<0.08) & (np.abs(year_comb['weight_kg']-w_kg)<12.0)]
    if len(cands) == 0: cands = year_comb
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

# For EACH training discrepancy, check if the matched player's name is in draft_picks
print("Training discrepancies - checking if matched player IS in draft_picks:")
print()
for _, row in train[train['Id'].isin(train_discrepancies)].iterrows():
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    name = best['player_name']
    year = int(row['Year'])
    
    # Check draft_picks by exact name
    dp_exact = draft_picks[(draft_picks['pfr_player_name'].str.lower() == name.lower()) &
                            (draft_picks['season'].between(year-1, year+1))]
    
    # Check draft_picks by last name
    last = name.split()[-1].lower()
    dp_last = draft_picks[(draft_picks['pfr_player_name'].str.lower().str.endswith(last)) &
                           (draft_picks['season'].between(year-1, year+1))]
    
    print(f"ID {int(row.Id):4d} | truth={int(row.Drafted)} | combine says drafted={bool(pd.notnull(best.draft_ovr))} | "
          f"matched='{name}' ({best.school}) | in_dp_exact={not dp_exact.empty}")
    if not dp_exact.empty:
        dp = dp_exact.iloc[0]
        print(f"       -> IN draft_picks: '{dp.pfr_player_name}' R{dp.round}P{dp.pick}")
    elif not dp_last.empty:
        print(f"       -> Similar last names in dp: {dp_last[['pfr_player_name','round','pick']].head(3).to_string()}")

# Now check: for the DISCREPANCY cases, what IS the actual combine match by name?
print()
print("\n\nFor the training discrepancy cases with truth=0:")
print("Their matched combine player has draft_ovr - let's verify by checking if that SPECIFIC")
print("combine player name matches draft_picks for the right year.")
print()

# Check specific discrepancy patterns
print("KEY INSIGHT CHECK: Are any of these players NOT actually in draft_picks?")
print("(i.e., combine has draft_ovr but the player was NOT actually drafted in reality?)")
print()
not_in_dp = []
for _, row in train[train['Id'].isin(train_discrepancies)].iterrows():
    result = get_best_match(row, combine_by_season)
    if result[0] is None: continue
    pen, best = result
    name = best['player_name']
    year = int(row['Year'])
    
    if not pd.notnull(best['draft_ovr']):
        continue  # Skip non-drafted combine matches
    
    dp_exact = draft_picks[(draft_picks['pfr_player_name'].str.lower() == name.lower()) &
                            (draft_picks['season'].between(year-1, year+2))]
    # Try name variations
    name_clean = name.lower().replace(' jr.','').replace(' jr','').replace(' iii','').replace(' ii','').strip()
    dp_clean = draft_picks[draft_picks['pfr_player_name'].str.lower().str.replace(' jr.','').str.replace(' jr','').str.replace(' iii','').str.replace(' ii','').str.strip() == name_clean]
    
    if dp_exact.empty and dp_clean.empty:
        not_in_dp.append({'Id': int(row.Id), 'truth': int(row.Drafted), 'name': name, 'year': year,
                          'school': best['school'], 'draft_ovr': best['draft_ovr']})
        print(f"ID {int(row.Id)} (truth={int(row.Drafted)}): '{name}' ({best.school}, draft_ovr={best.draft_ovr}) NOT in draft_picks!")

if not not_in_dp:
    print("All discrepancy matched players ARE in draft_picks.")
print()
print(f"\nTotal not in draft_picks: {len(not_in_dp)}")
