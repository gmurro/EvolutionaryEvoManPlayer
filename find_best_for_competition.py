import os
import pandas as pd
import numpy as np
from ast import literal_eval
import sys
from ea1.singlelayer_controller import PlayerController

sys.path.insert(0, "evoman")
from environment import Environment


LAYER_NODES = [20, 10, 5]
GAMES_FILE_NAME = "games_played.csv"
BEST_INDIVIDUAL_PATTERN = "best_individual_run_"

# enemies to play against
ENEMIES_TEST = [1, 2, 3, 4, 5, 6, 7, 8]

def init_env(path, layer_nodes):
    """
    Initializes environment for single objective mode with static enemy and ai player.
    """
    return


def play_game(env, best_individual):
    """
    Play a game against all enemies and return the individual gain of the execution.
    """
    fitness_list = []
    player_life_list = []
    enemy_life_list = []
    time_list = []

    for enemy in ENEMIES_TEST:
        # Update the enemy
        env.update_parameter('enemies', [enemy])

        fitness, player_life, enemy_life, time = env.play(best_individual)
        fitness_list.append(fitness)
        player_life_list.append(player_life)
        enemy_life_list.append(enemy_life)
        time_list.append(time)

    # return the gain against all enemies
    return sum(player_life_list) - sum(enemy_life_list)



# read all games files
games_files = []
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f == GAMES_FILE_NAME]:
        games_files.append(os.path.join(dirpath, filename))

# dict of best individuals for each EA and for each group of enemies trained on
all_bests = {}

for file in games_files:
    df_games = pd.read_csv(file, sep=";")
    # convert the element inside the column 'individual_gains' to an array
    df_games['individual_gains'] = df_games['individual_gains'].apply(literal_eval)

    # compute the mean of the individual_gains for each run
    df_games['individual_gains'] = df_games['individual_gains'].map(lambda x: np.array(x).mean())

    # find max of the means of the individual_gains
    id_max = df_games["individual_gains"].argmax()
    best_run = int(df_games.iloc[id_max]["n_run"])
    best_mean_gain = df_games.iloc[id_max]["individual_gains"]

    # store the n run of the best and its gain
    controller_best = os.path.join(os.path.dirname(file), BEST_INDIVIDUAL_PATTERN+ str(best_run)+".txt")
    all_bests[controller_best] = best_mean_gain

# find the best individual across all EAs
best_for_competition = max(all_bests, key=all_bests.get)
print(f"The best individual found is {best_for_competition} (mean of the gain = {round(all_bests[best_for_competition],2)})")

# initialize the environment
env = Environment(
        experiment_name=os.path.dirname(best_for_competition),
        playermode="ai",
        player_controller=PlayerController(LAYER_NODES[1]),
        speed="fastest",
        enemymode="static",
        level=2,
        logs="off",
        savelogs="no",
        sound="off",
        randomini="yes"
    )