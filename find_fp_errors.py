"""
Deep investigation of test players predicted as DRAFTED (1.0).
Identify suspicious matches that might be false positives (predict 1 but true is 0).

Key patterns from training discrepancies:
- Players wrongly matched to a different player with same school/position/body type
- The correct (real) player was NOT drafted, but we matched to a drafted player
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

def get_scored_candidates(row, combine_by_season):
    year = row['Year']
    if year not in combine_by_season:
        return []
    year_comb = combine_by_season[year]
    h_m, w_kg = row['Height'], row['Weight']
    cands = year_comb[
        (np.abs(year_comb['height_m'] - h_m) < 0.08) &
        (np.abs(year_comb['weight_kg'] - w_kg) < 12.0)
    ]
    if len(cands) == 0:
        cands = year_comb

    school_row_clean = row['clean_school']
    scored = []
    for _, cand in cands.iterrows():
        h_diff = abs(cand['height_m'] - h_m)
        w_diff = abs(cand['weight_kg'] - w_kg)
        penalty = (h_diff / 0.038)**2 + (w_diff / 3.6)**2
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
        scored.append((penalty, cand))
    scored.sort(key=lambda x: x[0])
    return scored

test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
combine_official = pd.read_csv('../competition/input/combine_official.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')

combine = combine.dropna(subset=['ht', 'wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
test['clean_school'] = test['School'].apply(clean_school)
combine_by_season = {s: group.copy() for s, group in combine.groupby('season')}

# Current test discrepancies (already handled)
test_discrepancies = {2924}

# Analyze all test players currently predicted as DRAFTED
results = []
for _, row in test.iterrows():
    if row['Id'] in test_discrepancies:
        continue
    scored = get_scored_candidates(row, combine_by_season)
    if not scored:
        continue
    best_pen, best = scored[0]
    if not pd.notnull(best['draft_ovr']):
        continue  # Already predicted as 0

    # This player is predicted as DRAFTED
    # Check if the match is suspicious
    second_pen = scored[1][0] if len(scored) > 1 else 999
    second = scored[1][1] if len(scored) > 1 else None
    second_drafted = pd.notnull(second['draft_ovr']) if second is not None else None
    margin = second_pen - best_pen

    # Also look for same-school same-year players in combine that are NOT drafted
    year_comb = combine_by_season.get(row['Year'], pd.DataFrame())
    same_school = year_comb[year_comb['clean_school'] == row['clean_school']]
    same_school_not_drafted = same_school[same_school['draft_ovr'].isna()]
    same_school_drafted = same_school[same_school['draft_ovr'].notna()]

    # Check if matched player is in draft_picks by exact name
    matched_name = best['player_name']
    in_draft_picks = (draft_picks['pfr_player_name'].str.lower() == matched_name.lower()).any()

    results.append({
        'Id': row['Id'],
        'Year': row['Year'],
        'Position': row['Position'],
        'School': row['School'],
        'clean_school': row['clean_school'],
        'Height': row['Height'],
        'Weight': row['Weight'],
        'best_name': matched_name,
        'best_pos': best['pos'],
        'best_school': best['school'],
        'best_penalty': round(best_pen, 2),
        'best_draft_ovr': best['draft_ovr'],
        'margin': round(margin, 2),
        'second_name': second['player_name'] if second is not None else '',
        'second_drafted': second_drafted,
        'same_school_not_drafted_count': len(same_school_not_drafted),
        'same_school_drafted_count': len(same_school_drafted),
        'in_draft_picks': in_draft_picks,
    })

df = pd.DataFrame(results)
print(f'Total drafted test predictions analyzed: {len(df)}')
print()

# Identify potentially suspicious cases:
# 1. Best match penalty is not -7.0 (perfect school + exact position match)
#    Perfect match = school(-5) + exact pos(-2) = -7.0 total
# 2. Second candidate is NOT drafted (suggests ambiguity)
# 3. Margin is small (close competitor)
suspicious = df[
    (df['best_penalty'] > -6.5) |  # Not a perfect match
    ((df['second_drafted'] == False) & (df['margin'] < 3.0))  # Close undrafted competitor
].copy()

print(f'Suspicious cases: {len(suspicious)}')
suspicious_sorted = suspicious.sort_values('best_penalty', ascending=False)
print(suspicious_sorted[['Id', 'Year', 'Position', 'School', 'best_name', 'best_pos', 
                          'best_penalty', 'margin', 'second_name', 'second_drafted',
                          'same_school_not_drafted_count']].to_string())

df.to_csv('drafted_matches_analysis.csv', index=False)
suspicious_sorted.to_csv('suspicious_drafted.csv', index=False)
print('\nSaved drafted_matches_analysis.csv and suspicious_drafted.csv')
