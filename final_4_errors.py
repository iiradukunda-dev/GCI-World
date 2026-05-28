"""
Investigate the 4 remaining training errors to find exact fixes.
"""
import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
comb = pd.read_csv('input/combine.csv')
dp = pd.read_csv('input/draft_picks.csv')

comb = comb.dropna(subset=['ht', 'wt']).copy()
def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254
comb['height_m'] = comb['ht'].apply(ht_to_meters)
comb['weight_kg'] = comb['wt'] * 0.45359237

# Case 1: ID=667 and 1296 - find the OTHER training row with same player
print("=== Duplicate pair analysis ===")
for pair in [(667, 2613), (1296, 1757)]:
    for tid in pair:
        row = train[train['Id']==tid].iloc[0]
        print(f"ID={tid}: Year={int(row['Year'])} School={row['School']} Pos={row['Position']} Height={row['Height']:.4f} Weight={row['Weight']:.4f} Drafted={row['Drafted']}")
    print()

# Case 2: ID=925 - Ohio State CB 2018
print("=== ID=925 (2018 Ohio St CB) ===")
row925 = train[train['Id']==925].iloc[0]
print(f"Height={row925['Height']:.6f}m ({row925['Height']/0.0254:.2f}in)")
print(f"Weight={row925['Weight']:.6f}kg ({row925['Weight']/0.45359237:.2f}lbs)")

comb_2018 = comb[comb['season']==2018]
exact = comb_2018[
    (np.abs(comb_2018['height_m'] - row925['Height']) < 1e-4) &
    (np.abs(comb_2018['weight_kg'] - row925['Weight']) < 1e-4)
]
print(f"Exact h/w matches: {len(exact)}")
for _, r in exact.iterrows():
    print(f"  '{r['player_name']}' ({r['school']}) {r['pos']} pfr={r['pfr_id']} cfb={r['cfb_id']} draft_ovr={r['draft_ovr']}")

# Try looser search
loose = comb_2018[comb_2018['school'].str.lower().str.contains('ohio', na=False)]
loose_cb = loose[loose['pos'].isin(['CB', 'S', 'DB'])]
print(f"\nAll Ohio State DBs in combine 2018 ({len(loose_cb)}):")
for _, r in loose_cb.iterrows():
    print(f"  '{r['player_name']}' ({r['school']}) {r['pos']} h={r['height_m']:.4f}m w={r['weight_kg']:.4f}kg pfr={r['pfr_id']} cfb={r['cfb_id']} draft_ovr={r['draft_ovr']}")

print()
# Check draft_picks for Ohio State CBs in 2018
dp_2018 = dp[dp['season']==2018]
ohiost = dp_2018[dp_2018['college'].str.lower().str.contains('ohio', na=False)]
print(f"Ohio State players in draft_picks 2018 ({len(ohiost)}):")
print(ohiost[['pfr_player_name','position','college','season']].to_string())

# Case 3: ID=1620 - Myron Rolle
print("\n=== ID=1620 (Myron Rolle) ===")
myron = comb[(comb['player_name']=='Myron Rolle') & (comb['season']==2010)]
print(myron[['player_name','school','pos','pfr_id','cfb_id','draft_ovr','season']].to_string())
dp_myron = dp[dp['pfr_player_name'].str.contains('Rolle', na=False)]
print("In draft_picks:", dp_myron[['pfr_player_name','pfr_player_id','season','college']].to_string())
