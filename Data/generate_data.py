import os
import time
import json
from nba_api.stats.endpoints import commonallplayers
from nba_api.stats.static import teams
from nba_api.stats.endpoints import LeagueGameFinder, BoxScoreAdvancedV2, BoxScoreSummaryV2, BoxScoreTraditionalV2
import pandas as pd


def is_home_team(team_id, home_team_id):
    return 1 if team_id == home_team_id else 0


def write_to_existing_csv(new_player_data, new_team_data, season):
    # Concatenate new player and team data
    pd_player_data = pd.concat(new_player_data, ignore_index=True)
    pd_team_data = pd.concat(new_team_data, ignore_index=True)

    output_folder = "saved_data"  # Folder to store intermediate results

    # Load existing data
    existing_player_data = pd.read_csv(os.path.join(output_folder, f"{season}_player_data.csv"))
    existing_team_data = pd.read_csv(os.path.join(output_folder, f"{season}_team_data.csv"))
    
    # Concatenate the new data with the existing data
    pd.concat([existing_player_data, pd_player_data], ignore_index=True).to_csv(
        os.path.join(output_folder, f"{season}_player_data.csv"), index=False
    )
    pd.concat([existing_team_data, pd_team_data], ignore_index=True).to_csv(
        os.path.join(output_folder, f"{season}_team_data.csv"), index=False
    )


def generate_save_data(season):
    # Fetch all games from the season
    with open(os.path.join('game_ids', f"game_ids_{season}.json"), "r") as f:
        game_ids = json.load(f)
    print(f'Number of games: {len(game_ids)}')

    output_folder = "saved_data"  # Folder to store intermediate results
    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(os.path.join(output_folder, f"{season}_player_data.csv")):
        data_exists = True
    else:
        data_exists = False

    new_player_data = []
    new_team_data = []

    times = []
    counter = 0
    save_interval = 100  # Save every 100 games

    try:
        while game_ids:
            game_id = game_ids.pop(0)
            with open(f"game_ids_{season}.json", "w") as f:
                json.dump(game_ids, f)

            start_time = time.time()

            boxscore_summary = BoxScoreSummaryV2(game_id=game_id)
            game_data = boxscore_summary.get_data_frames()[0]
            home_team_id = game_data['HOME_TEAM_ID'].iloc[0]
            date = game_data['GAME_DATE_EST'].iloc[0]

            boxscore = BoxScoreAdvancedV2(game_id=game_id)
            player_data = boxscore.get_data_frames()[0]
            team_data = boxscore.get_data_frames()[1]

            time.sleep(2)

            player_data['game_date'] = date
            team_data['game_date'] = date

            team_data['is_home'] = team_data['TEAM_ID'].apply(is_home_team, home_team_id=home_team_id)
            player_data['is_home'] = player_data['TEAM_ID'].apply(is_home_team, home_team_id=home_team_id)

            boxscore_traditional = BoxScoreTraditionalV2(game_id=game_id)
            player_traditional = boxscore_traditional.get_data_frames()[0]
            team_traditional = boxscore_traditional.get_data_frames()[1]

            new_player_cols = [col for col in player_traditional.columns if col not in player_data.columns]
            new_team_cols = [col for col in team_traditional.columns if col not in team_data.columns]

            for col in new_player_cols:
                player_data[col] = player_traditional[col]

            for col in new_team_cols:
                team_data[col] = team_traditional[col]

            new_player_data.append(player_data)
            new_team_data.append(team_data)

            counter += 1
            end_time = time.time()
            times.append(end_time - start_time)

            # Save data every 100 iterations
            if counter % save_interval == 0:
                if data_exists:
                    print(f"Checkpoint: Saving data at iteration {counter}")
                    # Append new player and team data to existing csv
                    write_to_existing_csv(new_player_data, new_team_data, season)
                    
                    # Reset the lists to hold the new batch of data
                    new_player_data.clear()
                    new_team_data.clear()
                else:
                    print(f"Checkpoint: Saving data at iteration {counter}")
                    # Write data to csv
                    pd.concat(new_player_data, ignore_index=True).to_csv(
                        os.path.join(output_folder, f"{season}_player_data.csv"), index=False
                    )
                    pd.concat(new_team_data, ignore_index=True).to_csv(
                        os.path.join(output_folder, f"{season}_team_data.csv"), index=False
                    )

            print(
                f'Remaining: {len(game_ids)}'
                f'\nEstimated time remaining: {round(((sum(times)/len(times)) * (len(game_ids))) / 60, 2)} mins'
            )
            time.sleep(1)

    except Exception as e:
        print(f"Exception occurred: {e}. Saving current progress...")
        if data_exists:
            # Append new player and team data to existing csv
            write_to_existing_csv(new_player_data, new_team_data, season)
        else:
            # Write data to csv
            pd.concat(new_player_data, ignore_index=True).to_csv(
                os.path.join(output_folder, f"{season}_player_data.csv"), index=False
            )
            pd.concat(new_team_data, ignore_index=True).to_csv(
                os.path.join(output_folder, f"{season}_team_data.csv"), index=False
            )
        raise

    # Final appending ot new player and team data after loop completion when game_ids is empty
    write_to_existing_csv(new_player_data, new_team_data, season)



if __name__ == "__main__":
    generate_save_data('2022-23')