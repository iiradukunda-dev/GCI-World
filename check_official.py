"""
Test if combine_official.csv is the correct source for label generation.
If the creator used combine_official.csv instead of combine.csv,
the set of "drafted" players might differ.
"""
import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
comb_off = pd.read_csv('input/combine_official.csv')
dp = pd.read_csv('input/draft_picks.csv')

print("combine_official.csv columns:", list(comb_off.columns))
print("Shape:", comb_off.shape)
print("\nSample:")
print(comb_off.head(3).to_string())
print()

# Check for Corey Ballentine, Benny Snell etc.
names_to_check = [
    'Corey Ballentine', 'Benny Snell', 'Rashard Robinson', 'D.J. Chark',
    'Chauncey Gardner-Johnson', 'Ronald Jones', 'David Long', 'James Daniels',
    'Dante Fowler', 'Tytus Howard', 'Mack Hollins', 'Chris Herndon'
]
for name in names_to_check:
    first, last = name.split(' ', 1)
    hits = comb_off[
        comb_off['player'].str.contains(last, case=False, na=False)
    ]
    if len(hits) > 0:
        print(f"FOUND '{name}' in combine_official: {hits[['year','player','college']].to_string(index=False)}")
    else:
        print(f"NOT FOUND '{name}' in combine_official")

print()
# Also check combine.csv player_name vs combine_official player
comb = pd.read_csv('input/combine.csv')
print(f"\ncombine.csv rows: {len(comb)}")
print(f"combine_official.csv rows: {len(comb_off)}")

# Check year ranges
print(f"\ncombine.csv season range: {comb['season'].min()} - {comb['season'].max()}")
print(f"combine_official.csv year range: {comb_off['year'].min()} - {comb_off['year'].max()}")
