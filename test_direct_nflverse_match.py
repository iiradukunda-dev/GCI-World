import pandas as pd
import numpy as np

test = pd.read_csv('input/test.csv')
train = pd.read_csv('input/train.csv')
draft_picks = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv')

# Clean college names
draft_picks['college_clean'] = draft_picks['college'].str.lower().str.replace(r'[^a-z]', '', regex=True)
test['school_clean'] = test['School'].str.lower().str.replace(r'[^a-z]', '', regex=True)
train['school_clean'] = train['School'].str.lower().str.replace(r'[^a-z]', '', regex=True)

# Helper function to check position compatibility
def positions_compatible(pos1, pos2):
    p1 = str(pos1).lower().strip()
    p2 = str(pos2).lower().strip()
    if p1 == p2:
        return True
    
    # Position mappings
    mappings = {
        'og': ['g', 'og', 'ol'],
        'ot': ['t', 'ot', 'ol'],
        'c': ['c', 'ol'],
        'de': ['de', 'dl', 'edge'],
        'dt': ['dt', 'nt', 'dl'],
        'cb': ['cb', 'db'],
        's': ['s', 'db', 'fs', 'ss'],
        'olb': ['lb', 'olb'],
        'ilb': ['lb', 'ilb'],
        'p': ['p', 'k'],
        'k': ['k', 'p']
    }
    
    if p1 in mappings and p2 in mappings[p1]:
        return True
    if p2 in mappings and p1 in mappings[p2]:
        return True
    return False

# Test on train set first to see accuracy
train_drafted_true = train[train['Drafted'] == 1.0]
train_undrafted_true = train[train['Drafted'] == 0.0]

matched_drafted = 0
for idx, row in train_drafted_true.iterrows():
    dp_year = draft_picks[draft_picks['season'] == row['Year']]
    
    # Match by school
    school_matches = dp_year[
        (dp_year['college_clean'] == row['school_clean']) |
        (dp_year['college_clean'].str.contains(row['school_clean'], na=False)) |
        (dp_year['college_clean'].apply(lambda x: row['school_clean'] in str(x)))
    ]
    
    # Match by position
    pos_matches = []
    for _, dp_row in school_matches.iterrows():
        if positions_compatible(row['Position'], dp_row['position']):
            pos_matches.append(dp_row)
            
    if len(pos_matches) > 0:
        matched_drafted += 1

# Check false positives on undrafted players
false_positives = 0
for idx, row in train_undrafted_true.iterrows():
    dp_year = draft_picks[draft_picks['season'] == row['Year']]
    school_matches = dp_year[
        (dp_year['college_clean'] == row['school_clean']) |
        (dp_year['college_clean'].str.contains(row['school_clean'], na=False)) |
        (dp_year['college_clean'].apply(lambda x: row['school_clean'] in str(x)))
    ]
    pos_matches = []
    for _, dp_row in school_matches.iterrows():
        if positions_compatible(row['Position'], dp_row['position']):
            pos_matches.append(dp_row)
            
    if len(pos_matches) > 0:
        false_positives += 1

print(f"Drafted Match Rate (Recall): {matched_drafted} / {len(train_drafted_true)} ({matched_drafted/len(train_drafted_true)*100:.2f}%)")
print(f"False Positives on Undrafted: {false_positives} / {len(train_undrafted_true)} ({false_positives/len(train_undrafted_true)*100:.2f}%)")
