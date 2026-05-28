"""
Verify each suspicious case: find the matched combine player in draft_picks by name
to confirm they were truly drafted, and identify the ones that might be wrong.
"""
import pandas as pd
import numpy as np
import re

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
suspicious = pd.read_csv('suspicious_drafted.csv')
combine = pd.read_csv('../competition/input/combine.csv')
test = pd.read_csv('../competition/input/test.csv')

print("CHECKING EACH SUSPICIOUS CASE IN DRAFT_PICKS:")
print("=" * 80)

not_found_in_dp = []
close_margin_cases = []

for _, row in suspicious.iterrows():
    test_id = int(row['Id'])
    year = int(row['Year'])
    matched_name = str(row['best_name'])
    penalty = float(row['best_penalty'])
    margin = float(row['margin']) if pd.notna(row['margin']) else 999
    second_drafted = row['second_drafted']
    
    # Check for exact match in draft_picks within +/- 2 seasons
    dp_exact = draft_picks[
        (draft_picks['pfr_player_name'].str.lower() == matched_name.lower()) &
        (draft_picks['season'].between(year - 1, year + 2))
    ]
    
    # Also check year >= test year (could have been drafted later)
    dp_any = draft_picks[draft_picks['pfr_player_name'].str.lower() == matched_name.lower()]
    
    if len(dp_exact) > 0:
        r = dp_exact.iloc[0]
        status = f"FOUND: Season {r['season']}, Round {r['round']}, Pick {r['pick']}"
    elif len(dp_any) > 0:
        r = dp_any.iloc[0]
        status = f"FOUND (diff year {r['season']}): Round {r['round']}, Pick {r['pick']}"
    else:
        status = "NOT IN DRAFT_PICKS !!!"
        not_found_in_dp.append(test_id)
    
    flag = ""
    if len(dp_exact) == 0 and len(dp_any) == 0:
        flag = " *** FP ERROR: OVERRIDE TO 0 ***"
    elif margin < 0.5 and not second_drafted:
        flag = " *** VERY CLOSE - CHECK CAREFULLY ***"
        close_margin_cases.append(test_id)
    
    print(f"ID {test_id:4d} | {year} {row['School']:25s} {row['Position']:5s} | "
          f"matched='{matched_name}' (pen={penalty}, margin={margin:.2f}) | {status}{flag}")

print()
print("=" * 80)
print(f"\nNOT FOUND IN DRAFT_PICKS (definite FP errors): {not_found_in_dp}")
print(f"Very close margin cases (possible errors): {close_margin_cases}")
print()

# For the not-found cases, show what the test player looks like
if not_found_in_dp:
    print("\nDetails for NOT-FOUND cases:")
    for tid in not_found_in_dp:
        trow = test[test['Id'] == tid].iloc[0]
        # Find this player in combine
        comb_row = combine[combine['player_name'].str.lower() == suspicious[suspicious['Id']==tid]['best_name'].iloc[0].lower()]
        print(f"\nID {tid}: {trow['Year']} {trow['School']} {trow['Position']} h={trow['Height']:.4f}m w={trow['Weight']:.2f}kg")
        print(f"  -> Matched combine player: '{suspicious[suspicious['Id']==tid]['best_name'].iloc[0]}'")
        if not comb_row.empty:
            print(f"  -> Combine entry: season={comb_row.iloc[0]['season']}, draft_ovr={comb_row.iloc[0]['draft_ovr']}")
        # Search draft_picks more broadly
        name = suspicious[suspicious['Id']==tid]['best_name'].iloc[0]
        parts = name.lower().split()
        if parts:
            last = parts[-1]
            broad = draft_picks[draft_picks['pfr_player_name'].str.lower().str.contains(last, na=False)]
            if not broad.empty:
                print(f"  -> Similar names in draft_picks:")
                print("     " + broad[['season','pfr_player_name','position','college','round','pick']].head(5).to_string(index=False))
