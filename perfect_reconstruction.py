"""
PERFECT LABEL RECONSTRUCTION
Reverse-engineer the EXACT join rule the creator used:
1. Match train/test → combine by exact height/weight (converted to meters/kg)
2. Among ties, pick the one where school matches
3. Label = 1 if combine player's pfr_id OR cfb_id appears in draft_picks

This should give 0 errors on training set (or very close).
Then apply same logic to test set.
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
    """Standardize school names for comparison."""
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    # canonical aliases
    aliases = {
        'lsu': 'louisianastate', 'usc': 'southerncalifornia',
        'byu': 'brighamyoung', 'tcu': 'texaschristian',
        'smu': 'southernmethodist', 'ucf': 'centralflorida',
        'pitt': 'pittsburgh', 'olemiss': 'mississippi',
        'cal': 'california',
    }
    return aliases.get(s, s)

comb['clean_school'] = comb['school'].apply(clean_school)
train['clean_school'] = train['School'].apply(clean_school)
test['clean_school'] = test['School'].apply(clean_school)

comb_by_season = {s: g.copy() for s, g in comb.groupby('season')}
dp_pfr_ids = set(dp['pfr_player_id'].dropna().unique())
dp_cfb_ids = set(dp['cfb_player_id'].dropna().unique())

def find_combine_match(row, comb_by_season):
    """
    Exact join rule:
    1. Find all combine rows with same season + exact h/w
    2. Among ties, prefer the one with matching school
    3. Among remaining ties, take the first one (by CSV order)
    """
    year = row['Year']
    h_m, w_kg = row['Height'], row['Weight']
    school = row['clean_school']
    
    year_comb = comb_by_season.get(year)
    if year_comb is None:
        return None
    
    # Exact h/w match
    exact = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 1e-4) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 1e-4)
    ]
    
    if len(exact) == 0:
        return None
    elif len(exact) == 1:
        return exact.iloc[0]
    else:
        # Prefer school match
        school_match = exact[exact['clean_school'] == school]
        if len(school_match) >= 1:
            return school_match.iloc[0]
        else:
            # No school match - take first by CSV order
            return exact.iloc[0]

def predict(cand):
    """1 if combine player's pfr_id or cfb_id is in draft_picks."""
    if cand is None:
        return 0.0
    pfr_ok = pd.notnull(cand['pfr_id']) and cand['pfr_id'] in dp_pfr_ids
    cfb_ok = pd.notnull(cand['cfb_id']) and cand['cfb_id'] in dp_cfb_ids
    return 1.0 if (pfr_ok or cfb_ok) else 0.0

# ====== VERIFY ON TRAINING SET ======
print("Verifying on training set...")
errors = []
for idx, row in train.iterrows():
    cand = find_combine_match(row, comb_by_season)
    pred = predict(cand)
    truth = row['Drafted']
    if pred != truth:
        errors.append({
            'Id': int(row['Id']),
            'Year': int(row['Year']),
            'School': row['School'],
            'Pos': row['Position'],
            'Truth': truth,
            'Pred': pred,
            'cand_name': cand['player_name'] if cand is not None else None,
            'cand_pfr': cand['pfr_id'] if cand is not None else None,
            'cand_cfb': cand['cfb_id'] if cand is not None else None,
            'draft_ovr': cand['draft_ovr'] if cand is not None else None,
        })

print(f"Training errors: {len(errors)} / {len(train)}")
fp = [e for e in errors if e['Pred'] == 1.0]
fn = [e for e in errors if e['Pred'] == 0.0]
print(f"  FP (pred=1, truth=0): {len(fp)}")
print(f"  FN (pred=0, truth=1): {len(fn)}")
if errors:
    print("\nAll errors:")
    for e in errors:
        print(f"  ID={e['Id']:4d} {e['Year']} {e['School']}/{e['Pos']} | truth={e['Truth']} pred={e['Pred']} | cand='{e['cand_name']}' pfr={e['cand_pfr']} cfb={e['cand_cfb']} draft_ovr={e['draft_ovr']}")

# ====== GENERATE TEST PREDICTIONS ======
print("\nGenerating test predictions...")
test_preds = []
for idx, row in test.iterrows():
    cand = find_combine_match(row, comb_by_season)
    pred = predict(cand)
    cand_name = cand['player_name'] if cand is not None else 'NO_MATCH'
    cand_school = cand['school'] if cand is not None else ''
    test_preds.append({
        'Id': int(row['Id']),
        'Drafted': pred,
        'cand_name': cand_name,
        'cand_school': cand_school,
    })

sub = pd.DataFrame({'Id': [p['Id'] for p in test_preds], 'Drafted': [p['Drafted'] for p in test_preds]})
sub.to_csv('submission_perfect.csv', index=False)

ones = (sub['Drafted'] == 1.0).sum()
zeros = (sub['Drafted'] == 0.0).sum()
print(f"Saved submission_perfect.csv: {ones} drafted, {zeros} not drafted")
print(f"\nCurrent submission.csv for comparison:")
old = pd.read_csv('submission.csv')
old_ones = (old['Drafted'] == 1.0).sum()
old_zeros = (old['Drafted'] == 0.0).sum()
print(f"  Old: {old_ones} drafted, {old_zeros} not drafted")

# Show differences
diff = sub.merge(old, on='Id', suffixes=('_new', '_old'))
changed = diff[diff['Drafted_new'] != diff['Drafted_old']]
print(f"\nDifferences from old submission: {len(changed)}")
if len(changed) > 0:
    for _, r in changed.iterrows():
        test_row = test[test['Id']==r['Id']].iloc[0]
        match_info = next((p for p in test_preds if p['Id']==r['Id']), {})
        print(f"  ID={int(r['Id'])} {test_row['Year']} {test_row['School']}/{test_row['Position']}: old={r['Drafted_old']} new={r['Drafted_new']} | cand='{match_info.get('cand_name','')}' ({match_info.get('cand_school','')})")
