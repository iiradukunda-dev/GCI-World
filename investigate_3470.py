"""
Deep investigate Test ID 3470 (2013 Fresno St. FS) and Training ID 334 (2012 Syracuse FS)
Both match 'Phillip Thomas' from combine.
Training ID 334 has truth=0 (not drafted).
What does combine say about Phillip Thomas?
"""
import pandas as pd
import numpy as np

train = pd.read_csv('../competition/input/train.csv')
test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237

# Phillip Thomas in combine
pt = combine[combine['player_name'].str.lower().str.contains('phillip thomas|philip thomas', na=False)]
print("Phillip Thomas in combine.csv:")
print(pt[['season','player_name','school','pos','ht','wt','draft_ovr']].to_string())
print()

# Training ID 334
tr334 = train[train['Id']==334].iloc[0]
print(f"Training ID 334: {tr334.Year} {tr334.School} {tr334.Position}")
print(f"  h={tr334.Height:.4f}m ({tr334.Height/0.0254:.3f}in), w={tr334.Weight:.4f}kg ({tr334.Weight/0.45359237:.3f}lbs)")
print(f"  Drafted={tr334.Drafted}")
print()

# Test ID 3470
te3470 = test[test['Id']==3470].iloc[0]
print(f"Test ID 3470: {te3470.Year} {te3470.School} {te3470.Position}")
print(f"  h={te3470.Height:.4f}m ({te3470.Height/0.0254:.3f}in), w={te3470.Weight:.4f}kg ({te3470.Weight/0.45359237:.3f}lbs)")
print()

# Phillip Thomas in draft_picks
pt_dp = draft_picks[draft_picks['pfr_player_name'].str.lower().str.contains('phillip thomas|philip thomas', na=False)]
print("Phillip Thomas in draft_picks:")
print(pt_dp[['season','pfr_player_name','position','college','round','pick']].to_string())
print()

# 2013 Fresno St. combine players
fs_2013 = combine[(combine['season']==2013) & (combine['school'].str.lower().str.contains('fresno', na=False))]
print("2013 Fresno State combine players:")
print(fs_2013[['player_name','pos','ht','wt','draft_ovr']].to_string())
print()

# 2012 Syracuse combine players
syr_2012 = combine[(combine['season']==2012) & (combine['school'].str.lower().str.contains('syracuse', na=False))]
print("2012 Syracuse combine players:")
print(syr_2012[['player_name','pos','ht','wt','draft_ovr']].to_string())

# Does Phillip Thomas appear in 2013 season?
pt_2013 = combine[(combine['player_name'].str.lower().str.contains('phillip thomas', na=False)) & (combine['season']==2013)]
print()
print("Phillip Thomas in 2013 combine:")
print(pt_2013.to_string())
