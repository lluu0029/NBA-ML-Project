import numpy as np
import pandas as pd


def generate_dataset(player_id, team_id, season):
    df_players = pd.read_csv(f'saved_data\{season}_player_data.csv')
    df_teams = pd.read_csv(f'saved_data\{season}_team_data.csv')

    df_players = df_players.drop_duplicates()
    df_teams = df_teams.drop_duplicates()

    df_player = df_players[df_players.PLAYER_ID == player_id]
    df_team = df_teams[df_teams.TEAM_ID != team_id]

    drop_team_cols = []

    # Convert MIN to float
    df_player['MIN'] = df_player['MIN'].apply(lambda x: round(float(x.split(':')[0]) + float(x.split(':')[1]) / 60, 1))

    # Order by game date, earliest to latest.
    df_player['game_date'] = pd.to_datetime(df_player['game_date'])
    df_player = df_player.sort_values(by='game_date')

    # Creating label 'NEXT_GAME_PTS
    df_player['NEXT_GAME_PTS'] = df_player.groupby('PLAYER_ID')['PTS'].shift(-1)

    # Days since last game
    df_player['LAST_GAME_DAYS'] = df_player.groupby('PLAYER_ID')['game_date'].diff().dt.days

    # Drop rows with NAN value
    df_player = df_player.dropna(subset=['NEXT_GAME_PTS', 'LAST_GAME_DAYS'])

    drop_player_cols = ['START_POSITION', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'PLAYER_ID', 'PLAYER_NAME', 'NICKNAME', 'COMMENT', 'game_date']
    df_player = df_player.drop(columns=drop_player_cols)

    numeric_cols = df_player.select_dtypes(include='number').columns
    numeric_cols = numeric_cols.drop('GAME_ID')

    # Calculate 2 day rolling mean 
    two_day_avg = (
        df_player[numeric_cols]
        .shift(1)  # shift by 1 to calculate from the 1st previous game
        .rolling(window=2, min_periods=2)
        .mean()
        .dropna()
        .reset_index(drop=True)
    )
    two_day_avg = two_day_avg.add_prefix('avg_last2_')

    two_three_ids = df_player['GAME_ID'].iloc[:2]
    # Filter the DataFrame to exclude those GAME_IDs
    two_day_game_ids = df_player[~df_player['GAME_ID'].isin(two_three_ids)]['GAME_ID']
    # Reset index before assigning GAME_IDs
    two_day_avg['GAME_ID'] = two_day_game_ids.reset_index(drop=True)


    # Calculate 3 day rolling mean 
    three_day_avg = (
        df_player[numeric_cols]
        .shift(1)  # shift by 1 to calculate from the 1st previous game
        .rolling(window=3, min_periods=3)
        .mean()
        .dropna()
        .reset_index(drop=True)
    )
    three_day_avg = three_day_avg.add_prefix('avg_last3_')

    first_three_ids = df_player['GAME_ID'].iloc[:3]
    # Filter the DataFrame to exclude those GAME_IDs
    three_day_game_ids = df_player[~df_player['GAME_ID'].isin(first_three_ids)]['GAME_ID']
    # Reset index before assigning GAME_IDs
    three_day_avg['GAME_ID'] = three_day_game_ids.reset_index(drop=True)
    # print(three_day_avg.head(5).to_dict())

    # Calculate 4 day rolling mean 
    four_day_avg = (
        df_player[numeric_cols]
        .shift(1)  # shift by 1 to calculate from the 1st previous game
        .rolling(window=4, min_periods=4)
        .mean()
        .dropna()
        .reset_index(drop=True)
    )
    four_day_avg = four_day_avg.add_prefix('avg_last4_')

    first_four_ids = df_player['GAME_ID'].iloc[:4]
    # Filter the DataFrame to exclude those GAME_IDs
    four_day_game_ids = df_player[~df_player['GAME_ID'].isin(first_four_ids)]['GAME_ID']
    # Reset index before assigning GAME_IDs
    four_day_avg['GAME_ID'] = four_day_game_ids.reset_index(drop=True)

    # Calculate 5 day rolling mean 
    five_day_avg = (
        df_player[numeric_cols]
        .shift(1)  # shift to exclude current game
        .rolling(window=5, min_periods=5)
        .mean()
        .dropna()
        .reset_index(drop=True)
    )
    five_day_avg = five_day_avg.add_prefix('avg_last5_')
    first_five_ids = df_player['GAME_ID'].iloc[:5]
    # Filter the DataFrame to exclude those GAME_IDs
    first_five_game_ids = df_player[~df_player['GAME_ID'].isin(first_five_ids)]['GAME_ID']
    # Reset index before assigning GAME_IDs
    five_day_avg['GAME_ID'] = first_five_game_ids.reset_index(drop=True)
    # print(five_day_avg.head(5).to_dict())
    # print(five_day_avg.columns.to_list())

    # print(len(five_day_avg))
    merged_df = five_day_avg.merge(df_player, on='GAME_ID', how='left')
    # print(len(merged_df))
    merged_df = merged_df.merge(four_day_avg, on='GAME_ID', how='left')
    merged_df = merged_df.merge(three_day_avg, on='GAME_ID', how='left')
    merged_df = merged_df.merge(two_day_avg, on='GAME_ID', how='left')
    # merged_df = merged_df.merge(five)
    # print(merged_df.columns.to_list())

    # print(df_player.columns.to_list())
    print(df_player.GAME_ID.head(6))
    print(df_player.PTS.head(6))
    # print(len(df_player))
    # print(len(three_day_avg))
    # print(len(five_day_avg))

    # print(merged_df.columns.to_list())
    print(merged_df.head(1)[['PTS', 'avg_last2_PTS', 'avg_last3_PTS', 'avg_last4_PTS', 'avg_last5_PTS', 'GAME_ID']])

    # print(merged_df[merged_df['GAME_ID'] == 22300139].to_dict())
    print(len(merged_df))
    merged_df = merged_df.drop('GAME_ID', axis=1)
    return merged_df


if __name__ == "__main__":
    name = 'Nikola Jokic'
    player_id = 203999
    # Denver Nuggets
    team_id = 1610612743
    season = '2023-24'

    df_player = generate_dataset(player_id, team_id, season)
    df_player.to_csv(rf'C:\Users\Lachlan\Documents\GitHub\NBA-ML-Project\datasets\{name}_{season}.csv', index=False)

