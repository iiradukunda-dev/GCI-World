"""
Look up the 10 combine players not found in draft_picks by exact name.
Find the correct name variant in draft_picks.
"""
import pandas as pd

draft_picks = pd.read_csv('../competition/input/draft_picks.csv')

# List of players to search for
players = [
    ('Jon Baldwin', 2011, 'WR'),
    ('Charles Leno', 2014, 'OG'),
    ('Jessie Bates', 2018, 'S'),
    ('Davon Gaudchaux', 2017, 'DT'),
    ('Trenton Brown', 2015, 'OG'),
    ('Vladimir Ducasse', 2010, 'OT'),
    ('Bisi Johnson', 2019, 'WR'),
    ('Matt Bosher', 2011, 'P'),
    ('Marvell Tell', 2019, 'S'),
    ('Mike Bennett', 2015, 'DT'),
]

for name, year, pos in players:
    # Try last name search
    last_name = name.split()[-1].lower()
    first_name = name.split()[0].lower()
    
    matches = draft_picks[
        (draft_picks['pfr_player_name'].str.lower().str.contains(last_name, na=False)) &
        (draft_picks['season'].between(year - 1, year + 1))
    ]
    
    print(f"'{name}' ({year}, {pos}):")
    if not matches.empty:
        print(matches[['season', 'pfr_player_name', 'position', 'college', 'round', 'pick']].to_string(index=False))
    else:
        # Try first name
        matches2 = draft_picks[
            (draft_picks['pfr_player_name'].str.lower().str.contains(first_name, na=False)) &
            (draft_picks['season'].between(year - 1, year + 1))
        ]
        if not matches2.empty:
            print(f"  (by first name): " + matches2[['season', 'pfr_player_name', 'position', 'college', 'round', 'pick']].to_string(index=False))
        else:
            # Try broader year range
            matches3 = draft_picks[
                draft_picks['pfr_player_name'].str.lower().str.contains(last_name, na=False)
            ]
            if not matches3.empty:
                print(f"  (any year): " + matches3[['season', 'pfr_player_name', 'position', 'college', 'round', 'pick']].head(5).to_string(index=False))
            else:
                print(f"  NOT FOUND AT ALL in draft_picks!")
    print()
