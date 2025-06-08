import numpy as np
import pandas as pd
import ast

def calc_avg_pts_per_min_teammates(row, df_players: pd.DataFrame):
    target_player_id = row.PLAYER_ID
    team_id = row.TEAM_ID
    game_date = row.game_date
    lineup = row.STARTING_PLAYER_IDs
    lineup = ast.literal_eval(lineup)   # Convert 'string' list back to type 'list'

    pts_per_min_avgs = []
    games_played_together = []

    for player_id in lineup:
        if player_id != target_player_id:
            # Dataframe for games including teammate
            teammate_games = df_players[df_players.PLAYER_ID == player_id]

            # GAME_IDs where that opponent played
            teammate_game_ids = set(teammate_games['GAME_ID'])

            # Filter games where target player played for their current team with their teammate
            played_together_games = df_players[
                (df_players.PLAYER_ID == target_player_id) &
                (df_players.TEAM_ID == team_id) &
                (df_players.GAME_ID.isin(teammate_game_ids))
            ]

            if len(played_together_games) > 0:
                # Convert MIN to float
                played_together_games['MIN'] = played_together_games['MIN'].apply(lambda x: round(float(x.split(':')[0]) + float(x.split(':')[1]) / 60, 1))
                # Order by game date, earliest to latest.
                played_together_games['game_date'] = pd.to_datetime(played_together_games['game_date'])
                played_together_games = played_together_games.sort_values(by='game_date')
                # Create feature pts/min
                played_together_games['PTS_PER_MIN'] = played_together_games['PTS'] / played_together_games['MIN']
                # Cap values for pts/min outliers, with min=0.4 and max=1.1
                played_together_games['PTS_PER_MIN'] = played_together_games['PTS_PER_MIN'].clip(lower=0.3)
                played_together_games['PTS_PER_MIN'] = played_together_games['PTS_PER_MIN'].clip(upper=1.3)

                # Filter rows to only those prior to the game_date
                played_together_games = played_together_games[played_together_games['game_date'] < game_date]

                # Calculate mean PTS_PER_MIN, add to list
                pts_per_min_avgs.append(played_together_games['PTS_PER_MIN'].mean())
                games_played_together.append(len(played_together_games))
                # print(played_together_games['PTS_PER_MIN'].to_list())
    
    # for i in range(len(pts_per_min_avgs)):
    #     print(f'pts_per_min_avg: {pts_per_min_avgs[i]}, games_played_together: {games_played_together[i]}')
    
    # Return avg pts/min across their starting lineup
    return sum(pts_per_min_avgs) / len(pts_per_min_avgs)


def calc_avg_pts_per_min_opponents(row, df_players: pd.DataFrame):
    target_player_id = row.PLAYER_ID
    team_id = row.TEAM_ID
    opp_team_id = row.OPP_TEAM_ID
    game_date = row.game_date
    lineup = row.OPP_PLAYER_IDs
    lineup = ast.literal_eval(lineup)   # Convert 'string' list back to type 'list'

    pts_per_min_avgs = []
    games_played_together = []

    for opp_player_id in lineup:
        if opp_player_id != target_player_id:
            # Dataframe for games including opponent player where they played for the opponent team
            opp_player_games = df_players[(df_players.PLAYER_ID == opp_player_id) & (df_players.TEAM_ID == opp_team_id)]

            # GAME_IDs where that opponent played
            opp_game_ids = set(opp_player_games['GAME_ID'])

            # Filter games where target player played for their current team and against the opponent player
            played_together_games = df_players[
                (df_players.PLAYER_ID == target_player_id) &
                (df_players.TEAM_ID == team_id) &
                (df_players.GAME_ID.isin(opp_game_ids))
            ]

            if len(played_together_games) > 0:
                # Convert MIN to float
                played_together_games['MIN'] = played_together_games['MIN'].apply(lambda x: round(float(x.split(':')[0]) + float(x.split(':')[1]) / 60, 1))
                # Order by game date, earliest to latest.
                played_together_games['game_date'] = pd.to_datetime(played_together_games['game_date'])
                played_together_games = played_together_games.sort_values(by='game_date')
                # Create feature pts/min
                played_together_games['PTS_PER_MIN'] = played_together_games['PTS'] / played_together_games['MIN']
                # Cap values for pts/min outliers, with min=0.4 and max=1.1
                played_together_games['PTS_PER_MIN'] = played_together_games['PTS_PER_MIN'].clip(lower=0.3)
                played_together_games['PTS_PER_MIN'] = played_together_games['PTS_PER_MIN'].clip(upper=1.3)

                # Filter rows to only those prior to the game_date
                played_together_games = played_together_games[played_together_games['game_date'] < game_date]

                # Calculate mean PTS_PER_MIN, add to list
                pts_per_min_avgs.append(played_together_games['PTS_PER_MIN'].mean())
                games_played_together.append(len(played_together_games))
                # print(played_together_games['PTS_PER_MIN'].to_list())
    
    # for i in range(len(pts_per_min_avgs)):
    #     print(f'pts_per_min_avg: {pts_per_min_avgs[i]}, games_played_together: {games_played_together[i]}')
    
    # Remove nan values if they exist
    pts_per_min_avgs = [x for x in pts_per_min_avgs if pd.notna(x)]

    # print(pts_per_min_avgs, sum(pts_per_min_avgs) / len(pts_per_min_avgs))

    # Data on 3 players required for it to be added
    if len(pts_per_min_avgs) >= 3:
        return sum(pts_per_min_avgs) / len(pts_per_min_avgs)    # Return avg pts/min across the opponent team lineup
    else:
        return np.nan   # Return nan, no data on opponents found


def add_seasonal_avgs(df, stat_cols: list):
    # Calculate averages for stat_cols
    season_avg = df[stat_cols].expanding().mean().shift(1)

    # Add prefix
    season_avg = season_avg.add_prefix('season_avg_')

    # Concatenate with original df
    return pd.concat([df, season_avg], axis=1)


def create_rolling_avg(df, stat_cols: list, number_games: int):
    df_rolling_avg = (
        df[stat_cols]
        .rolling(window=number_games, min_periods=number_games)
        .mean()
        .dropna()
        .reset_index(drop=True)
    )
    df_rolling_avg = df_rolling_avg.add_prefix(f'avg_last{number_games}_')

    # Add back original GAME_IDs
    df_rolling_avg['GAME_ID'] = df['GAME_ID'].iloc[number_games-1:].reset_index(drop=True)

    return df_rolling_avg


# def generate_dataset(player_id: int, team_id: int, season: str):
def generate_dataset(player_id: int, df_players, df_teams):
    df_players = df_players.drop_duplicates()
    df_teams = df_teams.drop_duplicates()

    df_player = df_players[df_players.PLAYER_ID == player_id].copy()
    df_team = df_teams[df_teams.TEAM_ID == team_id].copy()

    # Convert MIN to float
    df_player['MIN'] = df_player['MIN'].apply(lambda x: round(float(x.split(':')[0]) + float(x.split(':')[1]) / 60, 1))

    # Order by game date, earliest to latest.
    df_player['game_date'] = pd.to_datetime(df_player['game_date'])
    df_player = df_player.sort_values(by='game_date')

    # Create feature pts/min
    df_player['PTS_PER_MIN'] = df_player['PTS'] / df_player['MIN']

    # Cap values for pts/min outliers, with min=0.4 and max=1.1
    df_player['PTS_PER_MIN'] = df_player['PTS_PER_MIN'].clip(lower=0.3)
    df_player['PTS_PER_MIN'] = df_player['PTS_PER_MIN'].clip(upper=1.3)

    # # Creating label 'NEXT_GAME_PTS
    # df_player['NEXT_GAME_PTS'] = df_player.groupby('PLAYER_ID')['PTS'].shift(-1)
    # Creating label for the next game's pts/min
    df_player['NEXT_GAME_PTS_PER_MIN'] = (df_player.groupby('PLAYER_ID')['PTS_PER_MIN'].shift(-1))

    # Days since last game
    df_player['LAST_GAME_DAYS'] = df_player.groupby('PLAYER_ID')['game_date'].diff().dt.days
    # Days until next game
    df_player['DAYS_UNTIL_NEXT_GAME'] = df_player.groupby('PLAYER_ID')['LAST_GAME_DAYS'].shift(-1)


    # Shift is_home to create feature for whether next game is home
    df_player['NEXT_GAME_IS_HOME'] = df_player.groupby('PLAYER_ID')['is_home'].shift(-1)


    # Drop rows with NAN value
    # df_player = df_player.dropna(subset=['NEXT_GAME_PTS', 'LAST_GAME_DAYS'])
    df_player = df_player.dropna(subset=['NEXT_GAME_PTS_PER_MIN', 'LAST_GAME_DAYS'])


    stat_cols = ['MIN', 'E_OFF_RATING', 'OFF_RATING', 'E_DEF_RATING', 'DEF_RATING',
       'E_NET_RATING', 'NET_RATING', 'AST_PCT', 'AST_TOV', 'AST_RATIO',
       'OREB_PCT', 'DREB_PCT', 'REB_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT',
       'USG_PCT', 'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE',
       'is_home', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM',
       'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF',
       'PTS', 'PLUS_MINUS', 'DAYS_UNTIL_NEXT_GAME', 'PTS_PER_MIN']

    # Add seasonal avg stats
    df_player = add_seasonal_avgs(df_player, stat_cols)

    # DF for 5 game rolling avg
    five_game_rolling_avg = create_rolling_avg(df_player, stat_cols, 5)
    # DF for 3 game rolling avg
    # three_game_rolling_avg = create_rolling_avg(df_player, stat_cols, 3)

    # Merge rolling avgs with df_player
    df_player = five_game_rolling_avg.merge(df_player, on='GAME_ID', how='left')
    # df_player = df_player.merge(three_game_rolling_avg, on='GAME_ID', how='left')

    # Merge with df_team to include player team starting lineup
    df_player = df_player.merge(
        df_team[['GAME_ID', 'TEAM_ID', 'STARTING_PLAYER_IDs','STARTING_PLAYER_NAMEs']],
        on=['GAME_ID', 'TEAM_ID'],
        how='left'
    )

    # Relevant opponent team statistics 
    df_opponents = df_teams[['GAME_ID', 'TEAM_ID', 'TEAM_NAME', 'STARTING_PLAYER_IDs', 'STARTING_PLAYER_NAMEs']].copy()
    df_opponents = df_opponents.rename(columns={
        'TEAM_ID': 'OPP_TEAM_ID',
        'TEAM_NAME': 'OPP_TEAM_NAME',
        'STARTING_PLAYER_IDs': 'OPP_PLAYER_IDs',
        'STARTING_PLAYER_NAMEs': 'OPP_PLAYER_NAMEs'
    })

    # Merge statistics with df_player
    df_player = df_player.merge(df_opponents, on='GAME_ID', how='left')

    # Filter out own team labelled as opponents 
    df_player = df_player[df_player['TEAM_ID'] != df_player['OPP_TEAM_ID']].reset_index(drop=True)

    # Calculate avg pts/min with team lineup
    df_player['AVG_PTS_PER_MIN_WITH_TEAM'] = df_player.apply(lambda row: calc_avg_pts_per_min_teammates(row, df_players), axis=1)
    # Calculate avg pts/min with opponent lineup
    df_player['AVG_PTS_PER_MIN_AGAINST_OPP'] = df_player.apply(lambda row: calc_avg_pts_per_min_opponents(row, df_players), axis=1)

    # Shift avg pts/min for team and opponent lineup to create the avg pts/min for then next game.
    df_player['NEXT_AVG_PTS_PER_MIN_WITH_TEAM'] = df_player.groupby('PLAYER_ID')['AVG_PTS_PER_MIN_WITH_TEAM'].shift(-1)
    df_player['NEXT_AVG_PTS_PER_MIN_AGAINST_OPP'] = df_player.groupby('PLAYER_ID')['AVG_PTS_PER_MIN_AGAINST_OPP'].shift(-1)

    # Filter out 'AVG_PTS_PER_MIN_AGAINST_OPP' values which are nan
    df_player = df_player[df_player['NEXT_AVG_PTS_PER_MIN_AGAINST_OPP'].notna()]

    # Drop irrelevant columns
    drop_player_cols = [
        'GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'START_POSITION', 'COMMENT', 'game_date',
        'STARTING_PLAYER_IDs', 'STARTING_PLAYER_NAMEs', 'OPP_TEAM_ID', 'OPP_TEAM_NAME', 'OPP_PLAYER_IDs', 'OPP_PLAYER_NAMEs',
        'AVG_PTS_PER_MIN_WITH_TEAM', 'AVG_PTS_PER_MIN_AGAINST_OPP', 'LAST_GAME_DAYS', 'is_home'
    ]
    df_player = df_player.drop(columns=drop_player_cols)

    return df_player



if __name__ == "__main__":
    # Sample data
    name = 'Nikola Jokic'
    player_id = 203999  # Nikola Jokic
    team_id = 1610612743    # Denver Nuggets

    df_players_22 = pd.read_csv('saved_data/2022-23_player_data.csv')
    df_players_23 = pd.read_csv('saved_data/2023-24_player_data.csv')
    df_players_24 = pd.read_csv('saved_data/2024-25_player_data.csv')
    df_teams_22 = pd.read_csv('saved_data/2022-23_team_data.csv')
    df_teams_23 = pd.read_csv('saved_data/2023-24_team_data.csv')
    df_teams_24 = pd.read_csv('saved_data/2024-25_team_data.csv')

    df_players = pd.concat([df_players_22, df_players_23, df_players_24], ignore_index=True)
    df_teams = pd.concat([df_teams_22, df_teams_23, df_teams_24], ignore_index=True)

    df_player = generate_dataset(player_id, df_players, df_teams)

    # df_player.to_csv(f'datasets\{name}.csv', index=False)

    print(df_player[['NEXT_GAME_PTS_PER_MIN', 'PTS_PER_MIN', 'NEXT_AVG_PTS_PER_MIN_WITH_TEAM', 'NEXT_AVG_PTS_PER_MIN_AGAINST_OPP']].tail(10))
