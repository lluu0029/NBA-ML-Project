import json
import os
from nba_api.stats.endpoints import LeagueGameFinder

def generate_game_ids(season):
    game_finder = LeagueGameFinder(season_nullable=season, season_type_nullable='Regular Season')
    games = game_finder.get_data_frames()[0]

    # Extract the game IDs
    game_ids = games['GAME_ID'].unique().tolist()
    print(f'Number of games: {len(game_ids)}')

    if len(game_ids) == 0:
        print('Invalid season')
    else:
        output_folder = "game_ids"
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

        file_path = os.path.join(output_folder, f"game_ids_{season}.json")
        with open(file_path, "w") as f:
            json.dump(game_ids, f)


if __name__ == "__main__":
    generate_game_ids('2023-24')