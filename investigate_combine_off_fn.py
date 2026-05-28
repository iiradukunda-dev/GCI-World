"""
Investigate the 14 cases where we predict NOT DRAFTED (0) but combine_official
matches to a drafted player. Verify if these are truly errors or just
coincidental measurement matches (combine_official vs combine.csv have different values).
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
combine = pd.read_csv('../competition/input/combine.csv')
combine_off = pd.read_csv('../competition/input/combine_official.csv')
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
combine_off['height_m'] = combine_off['height'] * 0.0254
combine_off['weight_kg'] = combine_off['weight'] * 0.45359237

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

# The 14 suspicious cases
suspicious_ids = [2787, 2793, 2797, 2805, 2903, 3101, 3163, 3203, 3262, 3302, 3308, 3378, 3462, 3472]

print("INVESTIGATING 14 SUSPECTED FALSE NEGATIVES FROM COMBINE_OFFICIAL MATCHING")
print("=" * 100)

for test_id in suspicious_ids:
    trow = test[test['Id'] == test_id].iloc[0]
    year = int(trow['Year'])
    h_m = trow['Height']
    w_kg = trow['Weight']
    
    print(f"\nID {test_id}: {year} {trow['School']} {trow['Position']}")
    print(f"  Test: h={h_m:.4f}m ({h_m/0.0254:.3f}in), w={w_kg:.4f}kg ({w_kg/0.45359237:.3f}lbs)")
    
    # What combine_official matched to (within 0.003m and 0.5kg)
    off_match = combine_off[
        (combine_off['year'] == year) &
        (np.abs(combine_off['height_m'] - h_m) < 0.003) &
        (np.abs(combine_off['weight_kg'] - w_kg) < 0.5)
    ]
    
    print(f"  Combine_official matches (within 0.1in, 1lb):")
    for _, omatch in off_match.iterrows():
        dp = draft_picks[draft_picks['pfr_player_name'].str.lower() == omatch['player'].lower()]
        dp_info = f"R{dp.iloc[0]['round']}P{dp.iloc[0]['pick']}" if not dp.empty else "UDFA"
        print(f"    '{omatch['player']}' ({omatch['college']}, {omatch['position']}) "
              f"h={omatch['height']:.3f}in w={omatch['weight']:.1f}lbs | {dp_info}")
    
    # What combine.csv matches (our actual source)
    same_school_year = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == clean_school(trow['School']))
    ]
    
    print(f"  Combine.csv same-school players:")
    for _, cr in same_school_year.iterrows():
        h_diff = abs(cr['height_m'] - h_m) / 0.0254
        w_diff = abs(cr['weight_kg'] - w_kg) / 0.45359237
        dp = draft_picks[draft_picks['pfr_player_name'].str.lower() == cr['player_name'].lower()]
        dp_info = f"R{dp.iloc[0]['round']}P{dp.iloc[0]['pick']}" if not dp.empty else "UDFA"
        exact = "*** EXACT ***" if h_diff < 0.001 and w_diff < 0.001 else ""
        print(f"    '{cr['player_name']}' ({cr['pos']}) h={cr['ht']}/{cr['height_m']:.4f}m "
              f"(diff={h_diff:.3f}in) w={cr['wt']:.0f}/{cr['weight_kg']:.4f}kg "
              f"(diff={w_diff:.3f}lbs) draft_ovr={cr['draft_ovr']} | {dp_info} {exact}")
    
    # What combine_official match says the player's real name is
    if len(off_match) == 1:
        real_name = off_match.iloc[0]['player']
        real_college = off_match.iloc[0]['college']
        print(f"  >> combine_official identifies this player as: '{real_name}' ({real_college})")
        
        # Find this player in combine.csv
        in_combine = combine[combine['player_name'].str.lower() == real_name.lower()]
        if not in_combine.empty:
            ic = in_combine.iloc[0]
            print(f"  >> In combine.csv: {ic['player_name']} ({ic['school']}, {ic['pos']}) "
                  f"draft_ovr={ic['draft_ovr']} season={ic['season']}")
        else:
            # Try partial match
            last = real_name.split()[-1].lower()
            partial = combine[combine['player_name'].str.lower().str.contains(last, na=False) & 
                             (combine['season'] == year)]
            if not partial.empty:
                print(f"  >> Partial match in combine.csv:")
                for _, pr in partial.iterrows():
                    print(f"      '{pr['player_name']}' ({pr['school']}, {pr['pos']}) draft_ovr={pr['draft_ovr']}")
