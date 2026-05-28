"""
For ALL test players predicted as DRAFTED (1.0), verify the matched combine player
is in draft_picks.csv by name. If the matched player cannot be found in draft_picks
by name, this is suspicious (similar to ID 2801 which we checked before, but
Jon Baldwin appeared as 'Jonathan Baldwin').

Also look for ALL cases where combine has draft_ovr but the player's name
is NOT in draft_picks - these are likely the 25-style discrepancies.
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
combine = pd.read_csv('../competition/input/combine.csv')
test = pd.read_csv('../competition/input/test.csv')
sub = pd.read_csv('submission.csv')

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
test['clean_school'] = test['School'].apply(clean_school)
combine_by_season = {s: group.copy() for s, group in combine.groupby('season')}

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

# Build a full name lookup for draft_picks
dp_names = set(draft_picks['pfr_player_name'].str.lower().unique())

# For EACH combine player with draft_ovr, check if they're in draft_picks
combine_with_draft = combine[combine['draft_ovr'].notna()].copy()

not_in_dp = []
for _, cr in combine_with_draft.iterrows():
    name = cr['player_name'].lower()
    if name not in dp_names:
        # Try variations
        # Remove Jr, Sr, etc.
        name_clean = name.replace(' jr.', '').replace(' jr', '').replace(' sr.', '').replace(' sr', '')
        name_clean = name_clean.replace(' ii', '').replace(' iii', '').replace(' iv', '').strip()
        if name_clean not in dp_names:
            # Try swapping first/last name
            parts = name.split()
            if len(parts) == 2:
                swapped = f"{parts[1]} {parts[0]}"
                if swapped not in dp_names:
                    not_in_dp.append({
                        'combine_name': cr['player_name'],
                        'season': cr['season'],
                        'school': cr['school'],
                        'pos': cr['pos'],
                        'ht': cr['ht'],
                        'wt': cr['wt'],
                        'draft_ovr': cr['draft_ovr'],
                    })

df_not_in_dp = pd.DataFrame(not_in_dp)
print(f"Combine players with draft_ovr but NOT found in draft_picks by name: {len(df_not_in_dp)}")
print()
# Filter to test years (2009-2019)
test_years = df_not_in_dp[df_not_in_dp['season'].between(2009, 2019)]
print(f"Among test years (2009-2019): {len(test_years)}")
print()
print(test_years[['combine_name', 'season', 'school', 'pos', 'ht', 'wt', 'draft_ovr']].to_string())

# Now check: which of THESE combine players are the actual matches for our DRAFTED test predictions?
print("\n\nCross-referencing with test predictions:")
for _, cr in test_years.iterrows():
    # Find test player that matches this combine player
    year = int(cr['season'])
    h_m = ht_to_meters(cr['ht'])
    w_kg = float(cr['wt']) * 0.45359237 if pd.notna(cr['wt']) else np.nan
    cs = clean_school(cr['school'])
    
    if np.isnan(h_m) or np.isnan(w_kg):
        continue
    
    # Find matching test players
    test_matches = test[
        (test['Year'] == year) &
        (np.abs(test['Height'] - h_m) < 0.005) &
        (np.abs(test['Weight'] - w_kg) < 0.5)
    ]
    
    for _, tm in test_matches.iterrows():
        pred = sub_dict.get(tm['Id'], -1)
        print(f"  Combine: '{cr['combine_name']}' ({cr['school']}, {cr['pos']}, {cr['ht']}/{cr['wt']}) draft_ovr={cr['draft_ovr']}")
        print(f"  -> Test ID {int(tm['Id'])}: {tm['School']} {tm['Position']} h={tm['Height']:.4f} w={tm['Weight']:.4f} | PREDICTION={pred}")
        print()
