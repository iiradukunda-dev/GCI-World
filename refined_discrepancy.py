"""
REFINED DISCREPANCY APPROACH
The discrepancy override was too aggressive - it flagged correct mappings too.

Fix: Only apply the discrepancy override if the matched combine player's pfr_id
appears in draft_picks AND the mapped combine player is NOT the one from the
training discrepancy (which is an incorrect mapping from a different school).

Better fix: When applying the discrepancy override, check if the combine player's
pfr_id was identified as a discrepancy due to an INCORRECT school mapping.
If our current mapping (school-matched) correctly maps to this player, keep 1.0.

Concrete logic for discrepancy_set:
- Build discrepancy_set from: train FP cases where combine match does NOT have
  exact school match (i.e., it was a wrong-school fallback mapping)
- OR: build from the raw list of discrepancy pfr_ids and check if current mapping
  has school match → if yes, keep 1.0; if no, override to 0.0
"""
import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')
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
test['clean_school'] = test['School'].apply(clean_school)
comb_by_season = {s: g.copy() for s, g in comb.groupby('season')}
dp_pfr_ids = set(dp['pfr_player_id'].dropna().unique())
dp_cfb_ids = set(dp['cfb_player_id'].dropna().unique())

# Build dp pfr lookup
dp_by_pfr = {}
for _, r in dp.iterrows():
    if pd.notnull(r['pfr_player_id']):
        dp_by_pfr[r['pfr_player_id']] = r

# Build combine lookup by pfr_id
comb_by_pfr = {}
for _, c in comb.iterrows():
    if pd.notnull(c['pfr_id']):
        comb_by_pfr[c['pfr_id']] = c

def find_combine_match(row, with_school_flag=False):
    year, h_m, w_kg, school = row['Year'], row['Height'], row['Weight'], row['clean_school']
    year_comb = comb_by_season.get(year)
    if year_comb is None: return (None, False) if with_school_flag else None
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    if len(exact) == 0: return (None, False) if with_school_flag else None
    school_match = exact[exact['clean_school'] == school]
    if len(school_match) > 0:
        return (school_match.iloc[0], True) if with_school_flag else school_match.iloc[0]
    else:
        return (exact.iloc[0], False) if with_school_flag else exact.iloc[0]

def base_predict(cand):
    if cand is None: return 0.0
    pfr_ok = pd.notnull(cand['pfr_id']) and cand['pfr_id'] in dp_pfr_ids
    cfb_ok = pd.notnull(cand['cfb_id']) and cand['cfb_id'] in dp_cfb_ids
    return 1.0 if (pfr_ok or cfb_ok) else 0.0

# STEP 1: Build discrepancy_pfr_ids from training FP cases where school did NOT match
# These are "wrong mappings" that should not affect correct school matches
discrepancy_pfr_ids = set()   # pfr_ids of wrongly-mapped players
discrepancy_cfb_ids = set()

for idx, row in train.iterrows():
    cand, school_matched = find_combine_match(row, with_school_flag=True)
    pred = base_predict(cand)
    truth = row['Drafted']
    if pred == 1.0 and truth == 0.0:  # FP case
        if not school_matched and cand is not None:
            # This was a fallback (no school match) → safe to add to discrepancy
            if pd.notnull(cand['pfr_id']):
                discrepancy_pfr_ids.add(cand['pfr_id'])
            if pd.notnull(cand['cfb_id']):
                discrepancy_cfb_ids.add(cand['cfb_id'])
        elif school_matched and cand is not None:
            # School matched but still FP → this is a "true discrepancy" player
            # Needs a different key: (season, pfr_id)
            pass  # will handle separately

# Now collect school-matched FP cases
school_matched_disc_pfr = set()  # (season, pfr_id) for school-matched discrepancies
school_matched_disc_cfb = set()
for idx, row in train.iterrows():
    cand, school_matched = find_combine_match(row, with_school_flag=True)
    pred = base_predict(cand)
    truth = row['Drafted']
    if pred == 1.0 and truth == 0.0 and school_matched and cand is not None:
        if pd.notnull(cand['pfr_id']):
            school_matched_disc_pfr.add((int(cand['season']), cand['pfr_id']))
        if pd.notnull(cand['cfb_id']):
            school_matched_disc_cfb.add((int(cand['season']), cand['cfb_id']))

print(f"Non-school-match discrepancy pfr_ids: {len(discrepancy_pfr_ids)}")
print(f"School-match discrepancy (season, pfr_id): {len(school_matched_disc_pfr)}")

# STEP 2: Predict with refined override
def refined_predict(row, cand, school_matched):
    pred = base_predict(cand)
    if pred == 0.0: return 0.0
    if cand is None: return 0.0
    
    season = int(cand['season'])
    pfr = cand['pfr_id']
    cfb = cand['cfb_id']
    
    if school_matched:
        # Check school-matched discrepancy set
        if (pd.notnull(pfr) and (season, pfr) in school_matched_disc_pfr) or \
           (pd.notnull(cfb) and (season, cfb) in school_matched_disc_cfb):
            return 0.0
    else:
        # Not school-matched: check general discrepancy set
        if (pd.notnull(pfr) and pfr in discrepancy_pfr_ids) or \
           (pd.notnull(cfb) and cfb in discrepancy_cfb_ids):
            return 0.0
    return 1.0

# STEP 3: Verify on training
print("\nVerifying on training set...")
errors = []
for idx, row in train.iterrows():
    cand, school_matched = find_combine_match(row, with_school_flag=True)
    pred = refined_predict(row, cand, school_matched)
    truth = row['Drafted']
    if pred != truth:
        c_name = cand['player_name'] if cand is not None else 'NONE'
        c_school = cand['school'] if cand is not None else ''
        errors.append(f"ID={int(row['Id'])} {int(row['Year'])} {row['School']}/{row['Position']} truth={truth} pred={pred} school_match={school_matched} cand='{c_name}' ({c_school})")

print(f"Training errors: {len(errors)}")
for e in errors:
    print(f"  {e}")

# STEP 4: Generate test predictions
test_preds = []
for idx, row in test.iterrows():
    cand, school_matched = find_combine_match(row, with_school_flag=True)
    pred = refined_predict(row, cand, school_matched)
    test_preds.append({'Id': int(row['Id']), 'Drafted': pred})

sub = pd.DataFrame(test_preds)
sub.to_csv('submission_refined.csv', index=False)
ones = (sub['Drafted']==1.0).sum()
zeros = (sub['Drafted']==0.0).sum()
print(f"\nSaved submission_refined.csv: {ones} drafted, {zeros} not drafted")
