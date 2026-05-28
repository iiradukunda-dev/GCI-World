"""
Verify the 4 very-close margin cases and ID 2801 in detail.
For these, we need to determine which combine player the TEST player actually is.

Key cases to investigate:
- ID 2801: Jon Baldwin (Pittsburgh WR 2011) - NOT in draft_picks but has draft_ovr=26
- ID 3130: Vonn Bell (Ohio St FS 2016) vs Jalin Marshall - margin=0.46
- ID 2785: Anthony Averett (Alabama CB 2018) vs Levi Wallace - margin=0.06 !!!
- ID 3014: Stefon Diggs (Maryland WR 2015) vs Deon Long - margin=0.14 !!!
- ID 3137: Charles Scott (LSU RB 2010) vs Keiland Williams - margin=0.40
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
combine = pd.read_csv('../competition/input/combine.csv')
test = pd.read_csv('../competition/input/test.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str:
        return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

combine = combine.dropna(subset=['ht', 'wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237

cases = [2801, 3130, 2785, 3014, 3137]

for test_id in cases:
    print("=" * 80)
    trow = test[test['Id'] == test_id].iloc[0]
    print(f"ID {test_id}: {trow['Year']} {trow['School']} {trow['Position']}")
    print(f"  Physical: h={trow['Height']:.4f}m ({trow['Height']/0.0254:.1f}in), w={trow['Weight']:.2f}kg ({trow['Weight']/0.45359237:.1f}lbs)")
    print()
    
    # Get same school/year combine players
    year_school = combine[
        (combine['season'] == trow['Year']) &
        (combine['school'].str.lower().str.replace(' ', '') == trow['School'].lower().replace(' ', '').replace('&', '').replace('.', ''))
    ]
    
    if year_school.empty:
        # Try fuzzy school match
        year_school = combine[
            (combine['season'] == trow['Year']) &
            (combine['school'].str.lower().str.contains(trow['School'].lower()[:5], na=False))
        ]
    
    print(f"  Same school+year players in combine:")
    for _, cr in year_school.iterrows():
        h_diff = abs(cr['height_m'] - trow['Height'])
        w_diff = abs(cr['weight_kg'] - trow['Weight'])
        dist = (h_diff / 0.0254)  # inches diff
        wdist = (w_diff / 0.45359237)  # lbs diff
        dp_check = draft_picks[draft_picks['pfr_player_name'].str.lower() == cr['player_name'].lower()]
        dp_info = f"DRAFTED R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if not dp_check.empty else "not in dp"
        print(f"    {cr['player_name']:30s} pos={cr['pos']:5s} ht={cr['ht']:5s} ({dist:+.2f}in) wt={cr['wt']:5.0f} ({wdist:+.2f}lbs) draft_ovr={cr['draft_ovr']} | {dp_info}")
    
    # Also check test player exact measurements in combine
    exact = combine[
        (combine['season'] == trow['Year']) &
        (np.abs(combine['height_m'] - trow['Height']) < 0.005) &
        (np.abs(combine['weight_kg'] - trow['Weight']) < 0.5)
    ]
    if not exact.empty:
        print(f"\n  EXACT combine match by measurements:")
        for _, er in exact.iterrows():
            print(f"    {er['player_name']} ({er['school']}, {er['pos']}) draft_ovr={er['draft_ovr']}")
    print()

# Special check for Jon Baldwin
print("=" * 80)
print("Jon Baldwin in draft_picks check:")
# 2011 Pittsburgh WR - was he actually drafted?
baldwin = draft_picks[
    (draft_picks['pfr_player_name'].str.lower().str.contains('baldwin', na=False)) &
    (draft_picks['season'] == 2011)
]
print(baldwin[['season','pfr_player_name','position','college','round','pick']].to_string())
print()
# Also check combine directly
jon = combine[combine['player_name'].str.lower().str.contains('baldwin', na=False)]
print("Baldwin in combine.csv:")
print(jon[['season','player_name','school','pos','draft_ovr']].to_string())
