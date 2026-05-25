import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
pro_day = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_pro_day.csv')
draft_picks = pd.read_csv('https://github.com/nflverse/nflverse-data/releases/download/draft_picks/draft_picks.csv')

# Preprocess pro_day
pro_day = pro_day.dropna(subset=['Height (in)', 'Weight (lbs)']).copy()
pro_day['player_clean'] = pro_day['player'].str.lower().str.replace(r'[^a-z]', '', regex=True)
pro_day['college_clean'] = pro_day['College'].str.lower().str.replace(r'[^a-z]', '', regex=True)

# Preprocess draft_picks
draft_picks['player_clean'] = draft_picks['pfr_player_name'].str.lower().str.replace(r'[^a-z]', '', regex=True)

# School normalization dictionary
school_mapping = {
    'st': 'state',
    'oklahoma': 'oklahoma',
    'pittsburgh': 'pitt',
    'pitt': 'pittsburgh',
    'va': 'virginia',
    'virginia': 'va',
    'florida': 'fla',
    'mississippi': 'olemiss',
    'olemiss': 'mississippi',
    'california': 'cal',
    'cal': 'california',
    'bostoncollege': 'bc',
    'bc': 'bostoncollege',
    'texaschristian': 'tcu',
    'tcu': 'texaschristian',
    'southerncalifornia': 'usc',
    'usc': 'southerncalifornia',
    'northcarolinast': 'ncstate',
    'ncstate': 'northcarolinast',
    'byu': 'brighamyoung',
    'brighamyoung': 'byu',
    'lsu': 'louisianastate',
    'louisianastate': 'lsu',
    'smu': 'southernmethodist',
    'southernmethodist': 'smu'
}

def clean_school(name):
    if not isinstance(name, str):
        return ''
    s = name.lower().replace(' ', '').replace('.', '').replace('&', '').replace('-', '')
    for k, v in school_mapping.items():
        if s == k:
            return v
    return s

def school_match(s1, s2):
    cs1 = clean_school(s1)
    cs2 = clean_school(s2)
    if cs1 == cs2:
        return True
    if cs1 in cs2 or cs2 in cs1:
        return True
    return False

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
        ('rob', 'robert'), ('bobby', 'robert'), ('drew', 'andrew'),
        ('edmond', 'clyde'), ('tj', 'taylor'), ('danny', 'daniel'),
        ('geneo', 'gene'), ('pj', 'paul'), ('taven', 'tavyan'),
        ('robert', 'robby'), ('robby', 'robert')
    ]
    for n1, n2 in nickname_pairs:
        if (c_first == n1 and d_first == n2) or (c_first == n2 and d_first == n1):
            return True
    if len(c_first) >= 3 and len(d_first) >= 3:
        if c_first.startswith(d_first) or d_first.startswith(c_first):
            return True
    return False

def match_row(row, combine_df):
    h_inches = row['Height'] / 0.0254
    w_lbs = row['Weight'] / 0.45359237
    
    candidates = combine_df[combine_df['Year'] == row['Year']].copy()
    if len(candidates) == 0:
        return None
        
    cand_penalties = []
    
    for _, cand in candidates.iterrows():
        # Metric differences
        h_diff = abs(cand['Height (in)'] - h_inches)
        w_diff = abs(cand['Weight (lbs)'] - w_lbs)
        
        penalty = (h_diff / 1.5) ** 2 + (w_diff / 8.0) ** 2
        
        # Sprint
        if pd.notnull(cand['40 Yard']) and pd.notnull(row['Sprint_40yd']):
            penalty += ((cand['40 Yard'] - row['Sprint_40yd']) / 0.15) ** 2
            
        # Vert Leap
        if pd.notnull(cand['Vert Leap (in)']) and pd.notnull(row['Vertical_Jump']):
            penalty += ((cand['Vert Leap (in)'] - row['Vertical_Jump']/2.54) / 2.5) ** 2
            
        # Bench Press
        if pd.notnull(cand['Bench Press']) and pd.notnull(row['Bench_Press_Reps']):
            penalty += ((cand['Bench Press'] - row['Bench_Press_Reps']) / 4.0) ** 2
            
        # Broad Jump
        if pd.notnull(cand['Broad Jump (in)']) and pd.notnull(row['Broad_Jump']):
            penalty += ((cand['Broad Jump (in)'] - row['Broad_Jump']/2.54) / 6.0) ** 2
            
        # 3Cone
        if pd.notnull(cand['3Cone']) and pd.notnull(row['Agility_3cone']):
            penalty += ((cand['3Cone'] - row['Agility_3cone']) / 0.2) ** 2
            
        # Shuttle
        if pd.notnull(cand['Shuttle']) and pd.notnull(row['Shuttle']):
            penalty += ((cand['Shuttle'] - row['Shuttle']) / 0.2) ** 2
            
        # Position matching bonus
        pos_cand = str(cand['POS']).lower()
        pos_row = str(row['Position']).lower()
        if pos_cand == pos_row:
            penalty -= 1.0
        elif (pos_cand in ('og', 'ot', 'c', 'ol') and pos_row in ('og', 'ot', 'c', 'ol')) or \
             (pos_cand in ('wr', 'te', 'rb', 'qb') and pos_row in ('wr', 'te', 'rb', 'qb')) or \
             (pos_cand in ('dt', 'de', 'edge', 'dl') and pos_row in ('dt', 'de', 'edge', 'dl')) or \
             (pos_cand in ('cb', 's', 'db') and pos_row in ('cb', 's', 'db')) or \
             (pos_cand in ('olb', 'ilb', 'lb') and pos_row in ('olb', 'ilb', 'lb')):
            penalty -= 0.5
        else:
            penalty += 2.0
            
        cand_penalties.append(penalty)
        
    candidates['penalty'] = cand_penalties
    candidates = candidates.sort_values('penalty')
    best = candidates.iloc[0]
    
    # Strictly enforce school match and a penalty threshold
    if best['penalty'] < 5.0 and school_match(best['College'], row['School']):
        return best
    return None

errors = []
matched_count = 0

print("Evaluating strict matching logic on train.csv...")
for idx, row in train.iterrows():
    best_cand = match_row(row, pro_day)
    
    if best_cand is not None:
        matched_count += 1
        # Check draft status in draft_picks
        dp_year = draft_picks[draft_picks['season'] == row['Year']]
        is_drafted = 0.0
        for _, dp_row in dp_year.iterrows():
            if is_same_player(best_cand['player'], dp_row['pfr_player_name']):
                is_drafted = 1.0
                break
                
        # Compare with ground truth
        if is_drafted != row['Drafted']:
            errors.append((row['Id'], best_cand['player'], best_cand['College'], is_drafted, row['Drafted']))

print(f"\nStrictly matched {matched_count} out of {len(train)} players ({matched_count/len(train)*100:.2f}%)")
print(f"Number of errors: {len(errors)}")

if len(errors) > 0:
    print("\nErrors (Id, MatchedName, College, PredictedDrafted, ActualDrafted):")
    for err in errors[:30]:
        print(err)
