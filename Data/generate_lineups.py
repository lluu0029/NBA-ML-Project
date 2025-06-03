import pandas as pd

def generate_lineups(season: str):
    df_players = pd.read_csv(f'saved_data\{season}_player_data.csv')
    df_teams = pd.read_csv(f'saved_data\{season}_team_data.csv') 

    lineups_exist = {'STARTING_PLAYER_IDs', 'STARTING_PLAYER_NAMEs', 'STARTING_POSITIONs'}
    if lineups_exist.issubset(df_teams.columns):
        print('Lineups already exist')
        return

    # Filter out start_positions = nan  
    df_lineups = df_players[~df_players['START_POSITION'].isna()] 

    # Group by 'GAME_ID' and 'TEAM_ID', create list for PLAYER_IDs and PLAYER_NAMEs
    df_lineups = df_lineups.groupby(['GAME_ID', 'TEAM_ID']).agg({
        'PLAYER_ID': list,
        'PLAYER_NAME': list,
        'START_POSITION': list
    }).rename(columns={
        'PLAYER_ID': 'STARTING_PLAYER_IDs',
        'PLAYER_NAME': 'STARTING_PLAYER_NAMEs',
        'START_POSITION' : 'STARTING_POSITIONs'
    }).reset_index()

    # Add STARTING_PLAYER_IDs and STARTING_PLAYER_NAMEs columns to df_teams
    df_teams = df_teams.merge(df_lineups, on=['GAME_ID', 'TEAM_ID'], how='left')  

    # Update df_teams csv
    df_teams.to_csv(f'saved_data\\{season}_team_data.csv', index=False)
    print('Lineups created')

if __name__ == "__main__":
    generate_lineups('2024-25')