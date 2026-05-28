"""
Check ID 3070: USC WR 2017 - is this JuJu Smith-Schuster?
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

# ID 3070: USC WR 2017
trow = test[test['Id'] == 3070].iloc[0]
print('Test ID 3070:')
print(f'  Year={trow.Year}, School={trow.School}, Position={trow.Position}')
print(f'  Height={trow.Height:.4f}m ({trow.Height/0.0254:.2f}in), Weight={trow.Weight:.4f}kg ({trow.Weight/0.45359237:.2f}lbs)')
print()

# All USC 2017 combine players
usc_2017 = combine[
    (combine['season'] == 2017) & 
    (combine['school'].str.lower().str.replace(' ', '').str.contains('southerncal|usc', na=False))
]
print('USC 2017 combine players:')
for _, cr in usc_2017.iterrows():
    h_diff = abs(cr['height_m'] - trow.Height) / 0.0254
    w_diff = abs(cr['weight_kg'] - trow.Weight) / 0.45359237
    dp = draft_picks[draft_picks['pfr_player_name'].str.lower() == cr['player_name'].lower()]
    dp_info = f'R{dp.iloc[0]["round"]}P{dp.iloc[0]["pick"]}' if not dp.empty else 'UDFA'
    print(f'  {cr.player_name:30s} {cr.pos:5s} {cr.ht:5s}/{cr.wt:.0f} h_diff={h_diff:.1f}in w_diff={w_diff:.1f}lbs draft_ovr={cr.draft_ovr} | {dp_info}')

print()
# JuJu Smith-Schuster details
juju = combine[combine['player_name'].str.lower().str.contains('juju|smith', na=False)]
juju_usc = juju[juju['school'].str.lower().str.contains('southern|usc', na=False)]
print('JuJu/Smith in USC combine:')
print(juju_usc[['season', 'player_name', 'school', 'pos', 'ht', 'wt', 'draft_ovr']].to_string())

print()
juju_dp = draft_picks[draft_picks['pfr_player_name'].str.lower().str.contains('juju|schuster', na=False)]
print('JuJu in draft_picks:')
print(juju_dp[['season', 'pfr_player_name', 'position', 'college', 'round', 'pick']].to_string())

# Also: what is our current best match for ID 3070?
def clean_school(name):
    if not isinstance(name, str):
        return ''
    s = name.lower().replace(' ', '').replace('.', '').replace('&', '').replace('-', '').replace('(', '').replace(')', '')
    s = s.replace('university', '').replace('univ', '').replace('state', 'st')
    return s

combine['clean_school'] = combine['school'].apply(clean_school)
trow_cs = clean_school(trow.School)

year_comb = combine[combine['season'] == int(trow.Year)].copy()
cands = year_comb[
    (np.abs(year_comb['height_m'] - trow.Height) < 0.08) &
    (np.abs(year_comb['weight_kg'] - trow.Weight) < 12.0)
]

scored = []
for _, cand in cands.iterrows():
    h_diff = abs(cand['height_m'] - trow.Height)
    w_diff = abs(cand['weight_kg'] - trow.Weight)
    penalty = (h_diff / 0.038)**2 + (w_diff / 3.6)**2
    pos_cand = str(cand['pos']).lower()
    pos_row = trow.Position.lower()
    if pos_cand == pos_row:
        penalty -= 2.0
    elif pos_cand in ('wr', 'te', 'rb', 'qb', 'fb') and pos_row in ('wr', 'te', 'rb', 'qb'):
        penalty -= 1.0
    else:
        penalty += 2.0
    if cand['clean_school'] == trow_cs:
        penalty -= 5.0
    scored.append((penalty, cand))

scored.sort(key=lambda x: x[0])
print()
print('Our matching algorithm top 5 for ID 3070:')
for i, (pen, cand) in enumerate(scored[:5]):
    dp = draft_picks[draft_picks['pfr_player_name'].str.lower() == cand['player_name'].lower()]
    dp_info = f'R{dp.iloc[0]["round"]}P{dp.iloc[0]["pick"]}' if not dp.empty else 'UDFA'
    print(f'  [{i+1}] {cand.player_name:30s} ({cand.pos}, {cand.school}) pen={pen:.2f} draft_ovr={cand.draft_ovr} | {dp_info}')
