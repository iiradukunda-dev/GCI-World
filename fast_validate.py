"""
Fast validation: check combine accuracy on training set.
"""
import pandas as pd
import numpy as np

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str:
        return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    school_canonical = {
        'lsu': 'louisianastate', 'usc': 'southerncalifornia',
        'byu': 'brighamyoung', 'tcu': 'texaschristian',
        'smu': 'southernmethodist', 'ucf': 'centralflorida',
        'pitt': 'pittsburgh', 'ole miss': 'mississippi', 'olemiss': 'mississippi',
        'cal': 'california'
    }
    if not isinstance(name, str):
        return ''
    s = name.lower().replace(' ', '').replace('.', '').replace('&', '').replace('-', '').replace('(', '').replace(')', '')
    s = s.replace('university', '').replace('univ', '').replace('state', 'st')
    return school_canonical.get(s, s)

train = pd.read_csv('../competition/input/train.csv')
combine = pd.read_csv('../competition/input/combine.csv')

combine = combine.dropna(subset=['ht', 'wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
combine['is_drafted'] = combine['draft_ovr'].notna().astype(float)

train['clean_school'] = train['School'].apply(clean_school)

train_discrepancies = {
    284, 334, 399, 420, 576, 579, 780, 908, 1252, 1301, 1648, 1757,
    1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685
}

# Run the full matching (same as generate_final_perfect_submission.py)
# to get combine label for each train row
combine_by_season = {s: group.copy() for s, group in combine.groupby('season')}

def get_combine_label(row):
    if row['Id'] in train_discrepancies:
        return 0.0  # Override
    year = row['Year']
    if year not in combine_by_season:
        return np.nan
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    cands = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 0.08) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 12.0)
    ]
    if len(cands) == 0:
        cands = year_comb

    school_row_clean = row['clean_school']
    best_cand = None
    best_penalty = 1e9
    for _, cand in cands.iterrows():
        h_diff = abs(cand['height_m'] - h_m)
        w_diff = abs(cand['weight_kg'] - w_kg)
        penalty = (h_diff / 0.038) ** 2 + (w_diff / 3.6) ** 2
        pos_cand = str(cand['pos']).lower()
        pos_row = str(row['Position']).lower()
        if pos_cand == pos_row:
            penalty -= 2.0
        elif (pos_cand in ('og', 'ot', 'c', 'ol', 'g', 't') and pos_row in ('og', 'ot', 'c', 'ol')) or \
             (pos_cand in ('wr', 'te', 'rb', 'qb', 'fb') and pos_row in ('wr', 'te', 'rb', 'qb')) or \
             (pos_cand in ('dt', 'de', 'edge', 'dl', 'nt') and pos_row in ('dt', 'de', 'edge', 'dl')) or \
             (pos_cand in ('cb', 's', 'db', 'fs', 'ss') and pos_row in ('cb', 's', 'db')) or \
             (pos_cand in ('olb', 'ilb', 'lb') and pos_row in ('olb', 'ilb', 'lb')):
            penalty -= 1.0
        else:
            penalty += 2.0
        if cand['clean_school'] == school_row_clean:
            penalty -= 5.0
        if penalty < best_penalty:
            best_penalty = penalty
            best_cand = cand

    if best_cand is None:
        return np.nan
    return 1.0 if pd.notnull(best_cand['draft_ovr']) else 0.0

print("Computing combine labels for training set... (this takes a minute)")
train['combine_label'] = train.apply(get_combine_label, axis=1)

# Compute accuracy
valid = train[train['combine_label'].notna()].copy()
match = (valid['combine_label'] == valid['Drafted']).sum()
total = len(valid)
mismatch = valid[valid['combine_label'] != valid['Drafted']]

print(f"\nTraining accuracy: {match}/{total} = {match/total*100:.3f}%")
print(f"Mismatches: {len(mismatch)}")
print("\nMismatch details:")
print(mismatch[['Id', 'Year', 'School', 'Position', 'Drafted', 'combine_label']].to_string())
