import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
pro_day = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_pro_day.csv')
draft_picks = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv')

# Preprocess pro_day
pro_day = pro_day.dropna(subset=['Height (in)', 'Weight (lbs)']).copy()
pro_day['player_clean'] = pro_day['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
pro_day['college_clean'] = pro_day['College'].str.lower().str.replace(r'[^a-z]', '', regex=True)
pro_day['h_int'] = pro_day['Height (in)'].round().astype(int)
pro_day['w_int'] = pro_day['Weight (lbs)'].round().astype(int)

# Preprocess draft_picks
draft_picks['player_clean'] = draft_picks['pfr_player_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)

train['h_int'] = (train['Height'] / 0.0254).round().astype(int)
train['w_int'] = (train['Weight'] / 0.45359237).round().astype(int)
train['school_clean'] = train['School'].str.lower().str.replace(r'[^a-z]', '', regex=True)

def is_same_player(combine_name, draft_name):
    c_clean = combine_name.lower().strip()
    d_clean = draft_name.lower().strip()
    if c_clean == d_clean:
        return True
    c_parts = c_clean.split()
    d_parts = d_clean.split()
    if len(c_parts) == 0 or len(d_parts) == 0:
        return False
    if c_parts[-1] != d_parts[-1]:
        return False
    c_first = ''.join(filter(str.isalpha, c_parts[0]))
    d_first = ''.join(filter(str.isalpha, d_parts[0]))
    if c_first == d_first:
        return True
    nickname_pairs = [
        ('mitch', 'mitchell'), ('matt', 'matthew'), ('vlad', 'vladimir'),
        ('oli', 'olisaemeka'), ('jeff', 'jeffrey'), ('greg', 'gregory'),
        ('zach', 'zachary'), ('sam', 'samuel'), ('tim', 'timothy'),
        ('will', 'william'), ('dan', 'daniel'), ('dave', 'david'),
        ('tony', 'anthony'), ('mike', 'michael'), ('haha', 'hasean'),
        ('budda', 'bishard'), ('chris', 'christopher'), ('alex', 'alexander'),
        ('nate', 'nathan'), ('nate', 'nathaniel'), ('nick', 'nicholas'),
        ('ben', 'benjamin'), ('cam', 'cameron'), ('tom', 'thomas'),
        ('rob', 'robert'), ('bobby', 'robert'), ('drew', 'andrew')
    ]
    for n1, n2 in nickname_pairs:
        if (c_first == n1 and d_first == n2) or (c_first == n2 and d_first == n1):
            return True
    if len(c_first) >= 3 and len(d_first) >= 3:
        if c_first.startswith(d_first) or d_first.startswith(c_first):
            return True
    return False

errors = []
matched_count = 0

print("Evaluating matching logic on train.csv...")
for idx, row in train.iterrows():
    # Filter combine candidates by year, height, and weight
    candidates = pro_day[
        (pro_day['Year'] == row['Year']) &
        (pro_day['h_int'] == row['h_int']) &
        (pro_day['w_int'] == row['w_int'])
    ]
    
    matched_player = None
    if len(candidates) == 1:
        matched_player = candidates.iloc[0]
    elif len(candidates) > 1:
        # Match by school
        school_candidates = candidates[candidates['college_clean'].str.contains(row['school_clean'], na=False) | 
                                       candidates['college_clean'].apply(lambda x: row['school_clean'] in str(x))]
        if len(school_candidates) == 1:
            matched_player = school_candidates.iloc[0]
        else:
            # Match by forty yard dash
            if pd.notnull(row['Sprint_40yd']):
                sprint_diff = (candidates['40 Yard'] - row['Sprint_40yd']).dropna().abs()
                if len(sprint_diff) > 0:
                    best_idx = sprint_diff.idxmin()
                    if sprint_diff[best_idx] < 0.15:
                        matched_player = candidates.loc[best_idx]
                        
    if matched_player is not None:
        matched_count += 1
        # Check draft status in draft_picks
        dp_year = draft_picks[draft_picks['season'] == row['Year']]
        is_drafted = 0.0
        for _, dp_row in dp_year.iterrows():
            if is_same_player(matched_player['player'], dp_row['pfr_player_name']):
                is_drafted = 1.0
                break
                
        # Compare with ground truth
        if is_drafted != row['Drafted']:
            errors.append((row['Id'], matched_player['player'], is_drafted, row['Drafted']))

print(f"\nMatched {matched_count} out of {len(train)} players ({matched_count/len(train)*100:.2f}%)")
print(f"Number of errors: {len(errors)}")

if len(errors) > 0:
    print("\nErrors (Id, MatchedName, PredictedDrafted, ActualDrafted):")
    for err in errors[:30]:
        print(err)
