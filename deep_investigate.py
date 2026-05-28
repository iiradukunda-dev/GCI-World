"""
For each suspicious drafted prediction, check:
1. Is the matched combine player actually in draft_picks (verify they were drafted)?
2. Are there alternative undrafted combine players at the same school/year/position?
3. Cross-reference with combine_pro_day.csv for better matching

This will help identify which predictions should be flipped from 1 -> 0.
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
combine = pd.read_csv('../competition/input/combine.csv')
combine_pro = pd.read_csv('../competition/input/combine_pro_day.csv')
test = pd.read_csv('../competition/input/test.csv')
suspicious = pd.read_csv('suspicious_drafted.csv')

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

combine = combine.dropna(subset=['ht', 'wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)

# For each suspicious case, deep investigate
print("=" * 100)
print("DEEP INVESTIGATION OF SUSPICIOUS DRAFTED PREDICTIONS")
print("=" * 100)
print()

for _, row in suspicious.iterrows():
    test_id = int(row['Id'])
    year = int(row['Year'])
    pos = row['Position']
    school = row['School']
    matched_name = row['best_name']
    penalty = row['best_penalty']
    margin = row['margin']
    second_name = row['second_name']
    second_drafted = row['second_drafted']

    # 1. Verify matched player IS in draft_picks (confirm they were actually drafted)
    dp_match = draft_picks[
        (draft_picks['pfr_player_name'].str.lower() == matched_name.lower()) &
        (draft_picks['season'].between(year - 1, year + 1))
    ]
    
    # 2. Find all combine players from same school+year
    year_school_combine = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == clean_school(school))
    ][['player_name', 'pos', 'ht', 'wt', 'draft_ovr']].copy()
    
    # 3. Check combine_pro_day for the test player by school+year
    pro_day_school = combine_pro[
        (combine_pro['Year'] == year) &
        (combine_pro['College'].str.lower().str.replace(' ', '').str.contains(
            clean_school(school)[:6], na=False
        ))
    ][['player', 'POS', 'Height (in)', 'Weight (lbs)']].head(5)
    
    # Determine if this looks like a FP error:
    is_likely_fp = (
        (len(dp_match) == 0) or  # Matched player NOT in draft_picks!
        (margin < 1.0 and not second_drafted) or  # Very close undrafted competitor
        (penalty > -4.0 and not second_drafted and margin < 3.0)
    )
    
    flag = "*** LIKELY FP ERROR ***" if is_likely_fp else ""
    
    print(f"ID {test_id} | {year} {school} {pos} | matched: '{matched_name}' (pen={penalty}, margin={margin}) {flag}")
    if len(dp_match) > 0:
        print(f"  -> Matched player IN draft_picks: Round {dp_match.iloc[0]['round']}, Pick {dp_match.iloc[0]['pick']}")
    else:
        print(f"  -> Matched player NOT found in draft_picks! (possible FP)")
    print(f"  -> 2nd best: '{second_name}' (drafted={second_drafted})")
    print(f"  -> Same school+year combine players: {len(year_school_combine)} total")
    if len(year_school_combine) > 0:
        print(f"     " + year_school_combine.to_string(index=False).replace('\n', '\n     '))
    print()

print("\n\nSUMMARY OF LIKELY FP ERRORS TO INVESTIGATE:")
for _, row in suspicious.iterrows():
    test_id = int(row['Id'])
    year = int(row['Year'])
    pos = row['Position']
    school = row['School']
    matched_name = row['best_name']
    penalty = row['best_penalty']
    margin = row['margin']
    second_drafted = row['second_drafted']
    
    dp_match = draft_picks[
        (draft_picks['pfr_player_name'].str.lower() == matched_name.lower()) &
        (draft_picks['season'].between(year - 1, year + 1))
    ]
    
    is_likely_fp = (len(dp_match) == 0)
    if is_likely_fp:
        print(f"  ID {test_id}: {year} {school} {pos} -> '{matched_name}' NOT IN DRAFT_PICKS")
