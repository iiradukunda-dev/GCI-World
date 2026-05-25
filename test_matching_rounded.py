import pandas as pd
import numpy as np

train = pd.read_csv('input/train.csv')
combine = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_official.csv')

# Drop rows where height or weight is null in combine to avoid NaN errors
combine = combine.dropna(subset=['height', 'weight']).copy()
combine['h_int'] = combine['height'].round().astype(int)
combine['w_int'] = combine['weight'].round().astype(int)
combine['player_lower'] = combine['player'].str.lower().str.strip()

for i in range(10):
    row = train.iloc[i]
    h_inch = int(round(row['Height'] / 0.0254))
    w_lbs = int(round(row['Weight'] / 0.45359237))
    print(f"Train: School={row['School']} Year={row['Year']} Height={h_inch} Weight={w_lbs} Sprint={row['Sprint_40yd']}")
    matches = combine[
        (combine['year'] == row['Year']) &
        (combine['h_int'] == h_inch) &
        (combine['w_int'] == w_lbs)
    ]
    if len(matches) > 0:
        # Match by School/college or sprint time if there are multiple matches
        matched_row = None
        if len(matches) == 1:
            matched_row = matches.iloc[0]
        else:
            school = str(row['School']).lower().strip()
            # Try matching by school name
            school_matches = matches[matches['college'].str.lower().str.contains(school, na=False)]
            if len(school_matches) == 1:
                matched_row = school_matches.iloc[0]
            else:
                # Try matching by forty yard dash
                sprint_diff = (matches['forty_yard_dash'] - row['Sprint_40yd']).abs()
                best_idx = sprint_diff.idxmin()
                if sprint_diff[best_idx] < 0.1:
                    matched_row = matches.loc[best_idx]
        if matched_row is not None:
            print("Matched Player:", matched_row['player'], "College:", matched_row['college'])
        else:
            print("Could not disambiguate matches:")
            print(matches[['player', 'college', 'height', 'weight', 'forty_yard_dash']].to_string(index=False))
    else:
        print("No matches found")
    print('-'*50)
