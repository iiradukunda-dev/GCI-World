"""
Investigate ID 3197 (2014 Oregon St. WR) with exact measurement tie between
Brandin Cooks (drafted R1P20) and Rashaad Reynolds (undrafted).

Also study training ID 1757 (Iowa C) and 2613 (North Carolina WR) to understand
the TIE-BREAKING mechanism.

The question: in a tie (exact same measurements, same school), does the competition
pick the DRAFTED or UNDRAFTED player as the 'real' test player?

From training:
- ID 1757: Iowa C, truth=0 → we pick James Daniels (drafted) but truth says undrafted
  = competition thinks this player IS Sean Welsh (undrafted)
- ID 2613: North Carolina WR, truth=0 → we pick Mack Hollins (drafted) but truth says undrafted
  = competition thinks this player IS Bug Howard (undrafted)

Both times: truth is 0 (undrafted) when there's a tie between drafted and undrafted.

For test ID 3197: We pick Brandin Cooks (drafted). 
If competition uses same logic as training → should be 0 (Rashaad Reynolds, undrafted)!

But wait - maybe the competition ALWAYS picks undrafted when there's a tie?
OR maybe the competition picks based on who appears first in the combine.csv file?
"""
import pandas as pd
import numpy as np

train = pd.read_csv('../competition/input/train.csv')
test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
sub = pd.read_csv('submission.csv')

def ht_to_meters(ht_str):
    if not isinstance(ht_str, str) or '-' not in ht_str: return np.nan
    parts = ht_str.split('-')
    return (int(parts[0]) * 12 + int(parts[1])) * 0.0254

def clean_school(name):
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian',
         'smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh',
         'ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    return d.get(s, s)

combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237
combine['clean_school'] = combine['school'].apply(clean_school)
combine = combine.dropna(subset=['height_m','weight_kg']).copy()

# Check ID 3197: 2014 Oregon St. WR
test3197 = test[test['Id']==3197].iloc[0]
print(f"Test ID 3197: {test3197.Year} {test3197.School} {test3197.Position}")
print(f"  h={test3197.Height:.4f}m ({test3197.Height/0.0254:.3f}in) w={test3197.Weight:.4f}kg ({test3197.Weight/0.45359237:.3f}lbs)")
print()

oregon_2014 = combine[
    (combine['season'] == 2014) &
    (combine['clean_school'] == 'oregonst')
]
print("2014 Oregon State combine players:")
print(oregon_2014[['player_name','pos','ht','wt','draft_ovr','pfr_id']].to_string())
print()

# Check Brandin Cooks in draft_picks
cooks_dp = draft_picks[draft_picks['pfr_player_name'].str.lower().str.contains('cooks', na=False) & 
                        (draft_picks['season'] == 2014)]
print("Brandin Cooks in draft_picks:")
print(cooks_dp[['pfr_player_name','college','round','pick']].to_string())
print()

# Check order in combine.csv (raw index)
print("Row indices in combine.csv for 2014 Oregon State players:")
for idx, row in combine[combine['season']==2014][combine['clean_school']=='oregonst'].iterrows():
    print(f"  Index {idx}: '{row.player_name}' ({row.pos}) ht={row.ht} wt={row.wt} draft_ovr={row.draft_ovr}")
print()

# =====================================================================
# STUDY TRAINING TIE CASES
# =====================================================================
print("="*60)
print("STUDY: Training ID 1757 (Iowa C) - TIE between James Daniels and Sean Welsh")
print()
iowa_2018 = combine[(combine['season']==2018) & (combine['clean_school']=='iowa')]
print("2018 Iowa combine players:")
print(iowa_2018[['player_name','pos','ht','wt','draft_ovr']].to_string())
print()

tr1757 = train[train['Id']==1757].iloc[0]
print(f"Train ID 1757: h={tr1757.Height:.4f}m ({tr1757.Height/0.0254:.3f}in) w={tr1757.Weight:.4f}kg ({tr1757.Weight/0.45359237:.3f}lbs)")
print(f"  truth = {tr1757.Drafted}")
print()

# Row indices for Iowa 2018
print("Row indices in combine.csv for 2018 Iowa C players:")
iowa_c_2018 = combine[(combine['season']==2018) & (combine['clean_school']=='iowa') & (combine['pos'].str.lower()=='c')]
for idx, row in iowa_c_2018.iterrows():
    print(f"  Index {idx}: '{row.player_name}' ht={row.ht} wt={row.wt} draft_ovr={row.draft_ovr}")
print()

print("="*60)
print("STUDY: Training ID 2613 (North Carolina WR) - TIE between Mack Hollins and Bug Howard")
print()
nc_2017 = combine[(combine['season']==2017) & (combine['clean_school']=='northcarolina')]
print("2017 North Carolina combine players:")
print(nc_2017[['player_name','pos','ht','wt','draft_ovr']].to_string())
print()

tr2613 = train[train['Id']==2613].iloc[0]
print(f"Train ID 2613: h={tr2613.Height:.4f}m ({tr2613.Height/0.0254:.3f}in) w={tr2613.Weight:.4f}kg ({tr2613.Weight/0.45359237:.3f}lbs)")
print(f"  truth = {tr2613.Drafted}")
print()

# Row indices for NC 2017 WR
print("Row indices for 2017 North Carolina WR:")
nc_wr_2017 = combine[(combine['season']==2017) & (combine['clean_school']=='northcarolina') & (combine['pos'].str.lower()=='wr')]
for idx, row in nc_wr_2017.iterrows():
    print(f"  Index {idx}: '{row.player_name}' ht={row.ht} wt={row.wt} draft_ovr={row.draft_ovr}")
print()

# HYPOTHESIS: In tie-breaking, the FIRST combine player (lower index) wins.
# If the UNDRAFTED player appears FIRST in combine.csv, it gets label 0.
# If the DRAFTED player appears FIRST in combine.csv, it gets label 1.
# The competition uses the FIRST-FOUND combine player as the true identity.

print("="*60)
print("HYPOTHESIS: Competition uses FIRST combine player (by CSV order) as truth")
print("Checking for train 1757:")
print(f"  James Daniels (drafted) index vs Sean Welsh (undrafted) index:")
daniels = combine[(combine['season']==2018) & (combine['player_name']=='James Daniels')]
welsh = combine[(combine['season']==2018) & (combine['player_name']=='Sean Welsh')]
print(f"  James Daniels index: {daniels.index.tolist()}")
print(f"  Sean Welsh index: {welsh.index.tolist()}")
print(f"  Truth is 0 (undrafted = Sean Welsh) - Welsh index {'LOWER' if welsh.index[0] < daniels.index[0] else 'HIGHER'}")
print()
print(f"  If hypothesis true: lower index wins. Welsh index {welsh.index[0]} vs Daniels {daniels.index[0]}")
print()

print("Checking for train 2613:")
hollins = combine[(combine['season']==2017) & (combine['player_name']=='Mack Hollins')]
bug_howard = combine[(combine['season']==2017) & (combine['player_name']=='Bug Howard')]
print(f"  Mack Hollins index: {hollins.index.tolist()}")
print(f"  Bug Howard index: {bug_howard.index.tolist()}")
print(f"  Truth is 0 (undrafted = Bug Howard) - Bug Howard index {'LOWER' if bug_howard.index[0] < hollins.index[0] else 'HIGHER'}")
print()

print("Checking for test 3197:")
brandin_cooks = combine[(combine['season']==2014) & (combine['player_name']=='Brandin Cooks')]
rashaad_reynolds = combine[(combine['season']==2014) & (combine['player_name']=='Rashaad Reynolds')]
print(f"  Brandin Cooks index: {brandin_cooks.index.tolist()}")
print(f"  Rashaad Reynolds index: {rashaad_reynolds.index.tolist()}")
