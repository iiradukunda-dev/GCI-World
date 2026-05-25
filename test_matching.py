import pandas as pd

train = pd.read_csv('input/train.csv')
combine = pd.read_csv('https://raw.githubusercontent.com/array-carpenter/nfl-draft-data/master/data/combine_official.csv')

for i in range(5):
    row = train.iloc[i]
    h_inch = round(row['Height'] / 0.0254, 3)
    w_lbs = round(row['Weight'] / 0.45359237, 1)
    print(f"Train: School={row['School']} Year={row['Year']} Height={h_inch} Weight={w_lbs} Sprint={row['Sprint_40yd']}")
    matches = combine[
        (combine['year'] == row['Year']) &
        (abs(combine['height'] - h_inch) < 0.1) &
        (abs(combine['weight'] - w_lbs) < 0.5)
    ]
    print("Matches:")
    print(matches[['player', 'college', 'height', 'weight', 'forty_yard_dash']].to_string(index=False))
    print('-'*50)
