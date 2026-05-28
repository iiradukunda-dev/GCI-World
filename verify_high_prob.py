"""
Verify all high-probability uncertain players against draft_picks to confirm their draft status.
"""
import pandas as pd
import numpy as np

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
analysis = pd.read_csv('uncertain_analysis.csv')

high_prob = analysis[analysis['ml_pred'] > 0.5].copy()
print(f'High prob cases: {len(high_prob)}')

confirmed_drafted = []
confirmed_not_drafted = []

for _, row in high_prob.iterrows():
    name = str(row['match_name'])
    year = int(row['Year'])
    test_id = int(row['Id'])
    ml_pred = float(row['ml_pred'])
    
    # Exact name match
    exact = draft_picks[
        (draft_picks['pfr_player_name'].str.lower() == name.lower()) &
        (draft_picks['season'].between(year, year + 2))
    ]
    
    if len(exact) > 0:
        r = exact.iloc[0]
        confirmed_drafted.append({
            'Id': test_id, 'name': name, 'year': year, 'ml_pred': ml_pred,
            'draft_season': r['season'], 'round': r['round'], 'pick': r['pick']
        })
    else:
        confirmed_not_drafted.append({
            'Id': test_id, 'name': name, 'year': year, 'ml_pred': ml_pred
        })

print()
print('CONFIRMED DRAFTED (found in draft_picks by exact name match):')
for item in confirmed_drafted:
    print(f"  ID {item['Id']}: {item['name']} ({item['year']}) -> Round {item['round']}, Pick {item['pick']} (draft season {item['draft_season']})")

print()
print(f'CONFIRMED NOT DRAFTED ({len(confirmed_not_drafted)} players - undrafted FAs):')
for item in confirmed_not_drafted:
    print(f"  ID {item['Id']}: {item['name']} ({item['year']}) ml={item['ml_pred']:.3f}")
