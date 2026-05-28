"""Check ID 3064"""
import pandas as pd
import numpy as np

test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine = combine.dropna(subset=['height_m','weight_kg']).copy()

te = test[test['Id']==3064].iloc[0]
print(f'Test ID 3064: {te.Year} {te.School} {te.Position}')
print(f'  h={te.Height:.4f}m ({te.Height/0.0254:.3f}in) w={te.Weight:.4f}kg ({te.Weight/0.45359237:.3f}lbs)')
print()

sc_2014 = combine[(combine['season']==2014) & (combine['school'].str.lower().str.contains('south carolina', na=False))]
print('2014 South Carolina combine players:')
for _, cr in sc_2014.iterrows():
    h_diff = abs(cr['height_m'] - te.Height) / 0.0254
    w_diff = abs(cr['weight_kg'] - te.Weight) / 0.45359237
    exact = '*** EXACT ***' if h_diff < 0.001 and w_diff < 0.001 else ''
    print(f'  {cr.player_name:30s} {cr.pos:5s} {cr.ht:5s}/{cr.wt:.0f}  h_diff={h_diff:.3f}in  w_diff={w_diff:.3f}lbs  draft_ovr={cr.draft_ovr}  {exact}')

print()
vh = combine[(combine['player_name']=='Victor Hampton') & (combine['season']==2014)]
be = combine[(combine['player_name']=='Bruce Ellington') & (combine['season']==2014)]
print(f'Victor Hampton: ht={vh.iloc[0].ht} wt={vh.iloc[0].wt} h_m={vh.iloc[0].height_m:.4f} w_kg={vh.iloc[0].weight_kg:.4f} draft_ovr={vh.iloc[0].draft_ovr}')
print(f'Bruce Ellington: ht={be.iloc[0].ht} wt={be.iloc[0].wt} h_m={be.iloc[0].height_m:.4f} w_kg={be.iloc[0].weight_kg:.4f} draft_ovr={be.iloc[0].draft_ovr}')
print(f'Test player:     h_m={te.Height:.4f} w_kg={te.Weight:.4f}')
print()
print(f'Victor Hampton exact match to test: h={abs(vh.iloc[0].height_m - te.Height)/0.0254:.3f}in w={abs(vh.iloc[0].weight_kg - te.Weight)/0.45359237:.3f}lbs')
print(f'Bruce Ellington exact match to test: h={abs(be.iloc[0].height_m - te.Height)/0.0254:.3f}in w={abs(be.iloc[0].weight_kg - te.Weight)/0.45359237:.3f}lbs')
