"""
Use combine_pro_day.csv to cross-reference test players.
This file has player names AND measurements from NFL combine & pro days.
If a test player matches a combine_pro_day player exactly, we can get their name
and cross-check with draft_picks.

combine_pro_day.csv columns: Year, player, College, POS_GP, POS, Height (in), Weight (lbs), ...
"""
import pandas as pd
import numpy as np

test = pd.read_csv('../competition/input/test.csv')
combine = pd.read_csv('../competition/input/combine.csv')
combine_pro = pd.read_csv('../competition/input/combine_pro_day.csv')
draft_picks = pd.read_csv('../competition/input/draft_picks.csv')
sub = pd.read_csv('submission.csv')

print("combine_pro_day columns:", combine_pro.columns.tolist())
print("Sample:")
print(combine_pro.head(3).to_string())
print()
print(f"combine_pro_day years: {sorted(combine_pro.iloc[:,0].unique())[:15]}")

def clean_school(name):
    d = {'lsu':'louisianastate','usc':'southerncalifornia','byu':'brighamyoung','tcu':'texaschristian',
         'smu':'southernmethodist','ucf':'centralflorida','pitt':'pittsburgh',
         'ole miss':'mississippi','olemiss':'mississippi','cal':'california'}
    if not isinstance(name, str): return ''
    s = name.lower().replace(' ','').replace('.','').replace('&','').replace('-','').replace('(','').replace(')','')
    s = s.replace('university','').replace('univ','').replace('state','st')
    return d.get(s, s)

# Try to match test player to combine_pro by year+height+weight
# Need to convert test height/weight
# Test: Height in meters, Weight in kg
# combine_pro_day: Height in inches?, Weight in lbs?

# Find the year column and height/weight columns
year_col = combine_pro.columns[0]
print(f"\nYear column: '{year_col}', dtype={combine_pro[year_col].dtype}")

# Check what height values look like
height_col = [c for c in combine_pro.columns if 'height' in c.lower() or 'ht' in c.lower()]
weight_col = [c for c in combine_pro.columns if 'weight' in c.lower() or 'wt' in c.lower()]
print(f"Height columns: {height_col}")
print(f"Weight columns: {weight_col}")
print()
print("Sample height/weight values:")
print(combine_pro[[year_col] + height_col + weight_col].head(5).to_string())
