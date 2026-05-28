"""
Verify the most suspicious risky FP cases where BOTH players are drafted
(best AND alt are drafted) but we might have swapped them.

Also check: are there test players with TWO possible IDs (both drafted)?
In these cases, we predict 1.0 correctly regardless of which player we identify.

The TRUE false positives would only be cases where:
- Our top match is DRAFTED
- The 'actual' test player is the UNDRAFTED alternative

For training discrepancies:
- The training player is NOT drafted (truth=0)
- We match to a DRAFTED combine player
- This means the 'actual' combine player IS the training player BUT with draft_ovr NaN

Key insight: In the training data, the test/train player's measurements EXACTLY match
the combine UNDRAFTED player, but our algorithm picks the DRAFTED one because it has
a slightly better score (or same score but comes first in DataFrame).

Let me verify this for the training discrepancies:
For each training discrepancy ID, is there a combine player at that school with EXACTLY
matching measurements that is UNDRAFTED?
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
train['clean_school'] = train['School'].apply(clean_school)
test['clean_school'] = test['School'].apply(clean_school)

train_discrepancies = {284, 334, 399, 420, 576, 579, 780, 908, 1252, 1301, 1648, 1757,
                       1984, 1994, 2005, 2013, 2027, 2044, 2195, 2381, 2451, 2525, 2613, 2655, 2685}

print("FOR TRAINING DISCREPANCIES: Is there an EXACT match to an UNDRAFTED combine player?")
print("(This would confirm the true identity of the training player)")
print()
for _, row in train[train['Id'].isin(train_discrepancies)].iterrows():
    year = row['Year']
    cs = row['clean_school']
    h_m, w_kg = row['Height'], row['Weight']
    
    # Find all combine players at same school/year with exact measurements
    exact = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == cs) &
        (np.abs(combine['height_m'] - h_m) < 0.001) &
        (np.abs(combine['weight_kg'] - w_kg) < 0.01)
    ]
    
    if exact.empty:
        # No exact match - try looser
        close = combine[
            (combine['season'] == year) &
            (combine['clean_school'] == cs) &
            (np.abs(combine['height_m'] - h_m) < 0.005) &
            (np.abs(combine['weight_kg'] - w_kg) < 0.5)
        ]
        print(f"ID {int(row.Id):4d} ({row.Year} {row.School}): No exact school match! Close: {close[['player_name','draft_ovr']].to_string(index=False)}")
    else:
        for _, cr in exact.iterrows():
            print(f"ID {int(row.Id):4d} ({row.Year} {row.School}): EXACT match -> '{cr.player_name}' ({cr.pos}) draft={bool(pd.notnull(cr.draft_ovr))} draft_ovr={cr.draft_ovr}")

print()
print()

# Now: for each training discrepancy, the 'true' identity is the one with draft_ovr=NaN
# i.e., an undrafted player who happens to share measurements with a drafted player.
# But if there's only ONE exact match and it's the drafted player, then WHY is truth=0?
# Answer: The training player IS the drafted player but the COMPETITION chose not to label them as drafted!
# This would mean the COMPETITION uses a different labeling criterion.

# Let me check: are there test players whose measurements EXACTLY match an UNDRAFTED
# combine player from their school, but we match to a DRAFTED player (same school, similar measurements)?
print("="*80)
print("FOR TEST PLAYERS PREDICTED AS 1: Do they have an EXACT undrafted match at same school?")
print("(These would be high-confidence false positives)")
print()

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

possible_fp = []
for _, row in test.iterrows():
    test_id = row['Id']
    if sub_dict.get(test_id, 0) != 1.0: continue
    
    year = row['Year']
    cs = row['clean_school']
    h_m, w_kg = row['Height'], row['Weight']
    
    # What did we match? (the drafted player)
    drafted_match = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == cs) &
        (np.abs(combine['height_m'] - h_m) < 0.001) &
        (np.abs(combine['weight_kg'] - w_kg) < 0.01) &
        (combine['draft_ovr'].notna())
    ]
    
    # Is there also an UNDRAFTED player at same school/year with exact same measurements?
    undrafted_match = combine[
        (combine['season'] == year) &
        (combine['clean_school'] == cs) &
        (np.abs(combine['height_m'] - h_m) < 0.001) &
        (np.abs(combine['weight_kg'] - w_kg) < 0.01) &
        (combine['draft_ovr'].isna())
    ]
    
    if not undrafted_match.empty and not drafted_match.empty:
        possible_fp.append({
            'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
            'drafted_match': drafted_match.iloc[0]['player_name'],
            'undrafted_match': undrafted_match.iloc[0]['player_name'],
        })

print(f"Test predictions=1 with BOTH exact drafted AND undrafted matches: {len(possible_fp)}")
for fp in possible_fp:
    print(f"  ID {fp['Id']} {fp['Year']} {fp['School']} {fp['Position']}: "
          f"DRAFTED='{fp['drafted_match']}' vs UNDRAFTED='{fp['undrafted_match']}'")
