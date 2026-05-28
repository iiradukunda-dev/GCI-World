"""
Use pfr_id from combine.csv to directly match with draft_picks via pfr_player_id.
This is the most reliable way to verify if a combine player was actually drafted.

combine.csv has 'pfr_id' column.
draft_picks.csv has 'pfr_player_id' column.
These should match exactly!
"""
import pandas as pd
import numpy as np

train = pd.read_csv('../competition/input/train.csv')
test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
sub = pd.read_csv('submission.csv')

print("combine.csv pfr_id sample:")
print(combine[['player_name','school','season','pfr_id','draft_ovr']].head(10).to_string())
print()
print("draft_picks pfr_player_id sample:")
print(draft_picks[['pfr_player_name','pfr_player_id','season','college']].head(10).to_string())

# Verify: does pfr_id in combine match pfr_player_id in draft_picks?
dp_pfr_ids = set(draft_picks['pfr_player_id'].dropna().unique())
comb_drafted = combine[combine['draft_ovr'].notna()].copy()

# Count how many combine drafted players have pfr_id in draft_picks
comb_in_dp = comb_drafted[comb_drafted['pfr_id'].isin(dp_pfr_ids)]
comb_not_in_dp = comb_drafted[~comb_drafted['pfr_id'].isin(dp_pfr_ids)]
comb_null_pfr = comb_drafted[comb_drafted['pfr_id'].isna()]

print(f"\nCombine players with draft_ovr (drafted): {len(comb_drafted)}")
print(f"  -> pfr_id found in draft_picks: {len(comb_in_dp)}")
print(f"  -> pfr_id NOT found in draft_picks: {len(comb_not_in_dp[comb_not_in_dp['pfr_id'].notna()])}")
print(f"  -> pfr_id is NaN: {len(comb_null_pfr)}")
print()

# Show combine drafted players NOT in draft_picks (potential labeling errors)
not_matched = comb_not_in_dp[comb_not_in_dp['pfr_id'].notna()].copy()
test_years_range = set(test['Year'].unique())
not_matched_test_years = not_matched[not_matched['season'].isin(test_years_range)]
print(f"Combine drafted players with pfr_id NOT matching draft_picks (test years 2009-2019):")
print(not_matched_test_years[['season','player_name','school','pos','draft_ovr','pfr_id']].to_string())
