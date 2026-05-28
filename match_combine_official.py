"""
New approach: use combine_official.csv which has player NAMES + measurements.
Match each test player to combine_official to get their actual name,
then verify that name against draft_picks.csv directly.

This bypasses the school/position matching heuristic entirely.

combine_official has: year, player, college, position, height (in), weight (lbs)
test has: Year, School, Position, Height (m), Weight (kg)
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

# convert combine_official measurements to meters/kg
# height is in inches (decimal), weight in pounds
combine_off['height_m'] = combine_off['height'] * 0.0254
combine_off['weight_kg'] = combine_off['weight'] * 0.45359237

combine = combine.dropna(subset=['ht', 'wt']).copy()
combine['height_m'] = combine['ht'].apply(ht_to_meters)
combine['weight_kg'] = combine['wt'] * 0.45359237

sub_dict = dict(zip(sub['Id'], sub['Drafted']))

print("Combine_official columns:", combine_off.columns.tolist())
print("Sample combine_official:")
print(combine_off.head(3).to_string())
print()
print(f"Combine_official years: {sorted(combine_off['year'].unique())}")
print(f"Test years: {sorted(test['Year'].unique())}")
print()

# Try to match test players to combine_official by year + exact h/w
matches = []
for _, row in test.iterrows():
    test_id = row['Id']
    year = row['Year']
    h_m = row['Height']
    w_kg = row['Weight']
    
    # Find exact match in combine_official
    exact = combine_off[
        (combine_off['year'] == year) &
        (np.abs(combine_off['height_m'] - h_m) < 0.003) &   # within 0.1 inch
        (np.abs(combine_off['weight_kg'] - w_kg) < 0.5)     # within 1 lb
    ]
    
    n_exact = len(exact)
    if n_exact == 1:
        player_name = exact.iloc[0]['player']
        college = exact.iloc[0]['college']
        pos = exact.iloc[0]['position']
        
        # Verify in draft_picks
        dp_check = draft_picks[
            (draft_picks['pfr_player_name'].str.lower() == player_name.lower()) &
            (draft_picks['season'].between(year - 1, year + 1))
        ]
        in_dp = not dp_check.empty
        dp_info = f"R{dp_check.iloc[0]['round']}P{dp_check.iloc[0]['pick']}" if in_dp else "not_in_dp"
        
        current_pred = sub_dict.get(test_id, -1)
        
        matches.append({
            'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
            'matched_name': player_name, 'matched_college': college, 'matched_pos': pos,
            'in_draft_picks': in_dp, 'dp_info': dp_info,
            'current_pred': current_pred, 'n_exact': n_exact,
        })
    elif n_exact > 1:
        matches.append({
            'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
            'matched_name': f'AMBIGUOUS ({n_exact} matches)', 'matched_college': '',
            'matched_pos': '', 'in_draft_picks': None, 'dp_info': '',
            'current_pred': sub_dict.get(test_id, -1), 'n_exact': n_exact,
        })
    else:
        matches.append({
            'Id': test_id, 'Year': year, 'Position': row['Position'], 'School': row['School'],
            'matched_name': 'NO MATCH', 'matched_college': '',
            'matched_pos': '', 'in_draft_picks': None, 'dp_info': '',
            'current_pred': sub_dict.get(test_id, -1), 'n_exact': 0,
        })

df = pd.DataFrame(matches)
print(f"Total test players: {len(df)}")
print(f"Exact matches in combine_official: {(df['n_exact'] == 1).sum()}")
print(f"Ambiguous matches: {(df['n_exact'] > 1).sum()}")
print(f"No match: {(df['n_exact'] == 0).sum()}")
print()

# Focus on cases where combine_official name is found, check if prediction matches
exact_matches = df[df['n_exact'] == 1].copy()
print(f"Among exact matches ({len(exact_matches)} players):")

# Check for discrepancies: predicted as 1 but NOT in draft_picks
pred_1_not_dp = exact_matches[(exact_matches['current_pred'] == 1.0) & (~exact_matches['in_draft_picks'])]
print(f"  Predicted DRAFTED but NOT in draft_picks: {len(pred_1_not_dp)}")
if not pred_1_not_dp.empty:
    print(pred_1_not_dp[['Id', 'Year', 'School', 'Position', 'matched_name', 'matched_college', 'current_pred']].to_string())
print()

# Predicted as 0 but IS in draft_picks
pred_0_in_dp = exact_matches[(exact_matches['current_pred'] == 0.0) & (exact_matches['in_draft_picks'] == True)]
print(f"  Predicted NOT DRAFTED but IS in draft_picks: {len(pred_0_in_dp)}")
if not pred_0_in_dp.empty:
    print(pred_0_in_dp[['Id', 'Year', 'School', 'Position', 'matched_name', 'matched_college', 'dp_info', 'current_pred']].to_string())

df.to_csv('combine_official_matches.csv', index=False)
print("\nSaved combine_official_matches.csv")
