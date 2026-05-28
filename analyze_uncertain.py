"""
Analyze all uncertain (ML fallback) test players by tracing exact combine matching logic.
For players with high ML probability, verify actual drafted status.
"""
import pandas as pd
import numpy as np
import re

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str:
        return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    school_canonical = {
        'lsu': 'louisianastate', 'louisianastate': 'louisianastate',
        'usc': 'southerncalifornia', 'southerncalifornia': 'southerncalifornia',
        'byu': 'brighamyoung', 'brighamyoung': 'brighamyoung',
        'tcu': 'texaschristian', 'texaschristian': 'texaschristian',
        'smu': 'southernmethodist', 'southernmethodist': 'smu',
        'ucf': 'centralflorida', 'centralflorida': 'centralflorida',
        'pitt': 'pittsburgh', 'pittsburgh': 'pittsburgh',
        'ole miss': 'mississippi', 'olemiss': 'mississippi', 'mississippi': 'mississippi',
        'cal': 'california', 'california': 'california'
    }
    if not isinstance(name, str):
        return ''
    s = name.lower().replace(' ', '').replace('.', '').replace('&', '').replace('-', '').replace('(', '').replace(')', '')
    s = s.replace('university', '').replace('univ', '').replace('state', 'st')
    if s in school_canonical:
        return school_canonical[s]
    return s

def get_best_match(row, combine_by_season):
    year = row['Year']
    h_m = row['Height']
    w_kg = row['Weight']

    if year not in combine_by_season:
        return None, None

    year_comb = combine_by_season[year]
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

        if penalty < best_penalty:
            best_penalty = penalty
            best_cand = cand

    return best_cand, best_penalty

def main():
    test = pd.read_csv('../competition/input/test.csv')
    combine = pd.read_csv('../competition/input/combine.csv')
    draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
    sub = pd.read_csv('submission.csv')

    # Preprocess
    combine = combine.dropna(subset=['ht', 'wt']).copy()
    combine['height_m'] = combine['ht'].apply(ht_to_meters)
    combine['weight_kg'] = combine['wt'] * 0.45359237
    combine['clean_school'] = combine['school'].apply(clean_school)
    test['clean_school'] = test['School'].apply(clean_school)
    combine_by_season = {s: group.copy() for s, group in combine.groupby('season')}

    sub_dict = dict(zip(sub['Id'], sub['Drafted']))

    # Get uncertain (ML fallback) players
    uncertain = test[test['Id'].isin([iid for iid, v in sub_dict.items() if v < 1.0])].copy()

    print(f"Total uncertain players: {len(uncertain)}")
    print()

    results = []
    for _, row in uncertain.iterrows():
        iid = row['Id']
        ml_pred = sub_dict[iid]
        best_cand, best_penalty = get_best_match(row, combine_by_season)

        if best_cand is None:
            combine_status = 'NO_DATA'
            match_name = ''
            match_pos = ''
            match_school = ''
            is_drafted = None
        else:
            combine_status = 'DRAFTED' if pd.notnull(best_cand['draft_ovr']) else 'NOT_DRAFTED'
            match_name = best_cand['player_name']
            match_pos = best_cand['pos']
            match_school = best_cand['school']
            is_drafted = 1.0 if pd.notnull(best_cand['draft_ovr']) else 0.0

        results.append({
            'Id': iid,
            'Year': row['Year'],
            'Position': row['Position'],
            'School': row['School'],
            'ml_pred': ml_pred,
            'match_name': match_name,
            'match_pos': match_pos,
            'match_school': match_school,
            'penalty': round(best_penalty, 2) if best_penalty != 1e9 else None,
            'combine_status': combine_status,
            'combine_drafted': is_drafted,
        })

    df = pd.DataFrame(results)

    # Show summary
    print("Combine status distribution:")
    print(df['combine_status'].value_counts())
    print()

    # CASES WHERE COMBINE SAYS DRAFTED (but ML has low/medium prediction)
    # These are potential bugs - if combine match is good quality, should be 1.0
    drafted_cases = df[df['combine_status'] == 'DRAFTED'].copy()
    print(f"Cases where combine match shows DRAFTED: {len(drafted_cases)}")
    print(drafted_cases[['Id', 'Year', 'Position', 'School', 'ml_pred', 'match_name', 'match_pos', 'penalty']].to_string())
    print()

    # Cases where ML pred > 0.5 but combine says NOT_DRAFTED
    high_not_drafted = df[(df['ml_pred'] > 0.5) & (df['combine_status'] == 'NOT_DRAFTED')].copy()
    print(f"Cases ML>0.5 but combine says NOT_DRAFTED: {len(high_not_drafted)}")
    print(high_not_drafted[['Id', 'Year', 'Position', 'School', 'ml_pred', 'match_name', 'match_pos', 'penalty']].to_string())

    df.to_csv('uncertain_analysis.csv', index=False)
    print()
    print("Saved uncertain_analysis.csv")

if __name__ == '__main__':
    main()
