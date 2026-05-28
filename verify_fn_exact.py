"""
Deep verification of top FN candidates.
For each, check exact measurements in combine vs test to determine
which combine player the test player actually is.
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

fn_cases = pd.read_csv('fn_candidates.csv')

print("DETAILED VERIFICATION OF NEAR-MISS FALSE NEGATIVES")
print("=" * 100)

for _, fn in fn_cases.iterrows():
    test_id = int(fn['Id'])
    year = int(fn['Year'])
    
    trow = test[test['Id'] == test_id].iloc[0]
    
    print(f"\nID {test_id}: {year} {fn['School']} {fn['Position']}")
    print(f"  Test: ht={trow.Height:.4f}m ({trow.Height/0.0254:.3f}in), wt={trow.Weight:.4f}kg ({trow.Weight/0.45359237:.3f}lbs)")
    
    # Get chosen player exact measurements
    chosen = combine[combine['player_name'].str.lower() == fn['chosen_name'].lower()]
    if not chosen.empty:
        cr = chosen.iloc[0]
        h_diff = abs(cr['height_m'] - trow.Height)
        w_diff = abs(cr['weight_kg'] - trow.Weight)
        print(f"  CHOSEN (not_drafted): '{fn['chosen_name']}' ht={cr.ht} ({cr.height_m:.4f}m, diff={h_diff/0.0254:.3f}in) wt={cr.wt:.0f} ({cr.weight_kg:.4f}kg, diff={w_diff/0.45359237:.3f}lbs)")
    
    # Get alternative player exact measurements
    alt = combine[combine['player_name'].str.lower() == fn['alt_name'].lower()]
    if not alt.empty:
        ar = alt.iloc[0]
        h_diff2 = abs(ar['height_m'] - trow.Height)
        w_diff2 = abs(ar['weight_kg'] - trow.Weight)
        print(f"  ALT   (drafted={fn['alt_draft_ovr']} {fn['alt_in_dp']}): '{fn['alt_name']}' ht={ar.ht} ({ar.height_m:.4f}m, diff={h_diff2/0.0254:.3f}in) wt={ar.wt:.0f} ({ar.weight_kg:.4f}kg, diff={w_diff2/0.45359237:.3f}lbs)")
    
    print(f"  MARGIN: {fn['margin']:.4f} penalty units")
    
    # Key question: Which measurement is EXACTLY right?
    exact_chosen = not chosen.empty and abs(chosen.iloc[0]['height_m'] - trow.Height) < 0.001 and abs(chosen.iloc[0]['weight_kg'] - trow.Weight) < 0.01
    exact_alt = not alt.empty and abs(alt.iloc[0]['height_m'] - trow.Height) < 0.001 and abs(alt.iloc[0]['weight_kg'] - trow.Weight) < 0.01
    
    if exact_chosen:
        print(f"  >>> EXACT MATCH to CHOSEN (undrafted) player - prediction 0 is CORRECT")
    if exact_alt:
        print(f"  >>> EXACT MATCH to ALT (drafted) player - prediction 0 is WRONG, should be 1")
    if not exact_chosen and not exact_alt:
        print(f"  >>> No exact match - ambiguous case")

print()
print("=" * 100)
print("\nSUMMARY:")
print("Cases where test player EXACTLY matches the drafted alternative:")
for _, fn in fn_cases.iterrows():
    test_id = int(fn['Id'])
    trow = test[test['Id'] == test_id].iloc[0]
    alt = combine[combine['player_name'].str.lower() == fn['alt_name'].lower()]
    if not alt.empty:
        ar = alt.iloc[0]
        h_diff = abs(ar['height_m'] - trow.Height)
        w_diff = abs(ar['weight_kg'] - trow.Weight)
        if h_diff < 0.001 and w_diff < 0.01:
            print(f"  ID {test_id}: '{fn['alt_name']}' {fn['alt_in_dp']} (SHOULD BE DRAFTED=1)")
