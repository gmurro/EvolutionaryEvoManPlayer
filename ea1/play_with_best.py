import sys, os
import pandas as pd

sys.path.insert(0, "evoman")
from environment import Environment
from singlelayer_controller import PlayerController

# imports other libs
import numpy as np
import glob
import re

# group of enemies trained on
ENEMIES_TRAINING = [2, 5, 8]

# enemies to play against
ENEMIES_TEST = [1, 2, 3, 4, 5, 6, 7, 8]

RUNS_DIR = "runs"
N_GAMES = 5
BEST_INDIVIDUAL_PATTERN = "best_individual_run_"

# Update the number of neurons for the controller used
LAYER_NODES = [20, 10, 5]


def enemies_dir_name(enemies):
    """
    Converts a list of enemies into a string
    """
    dir_name = ""
    for enemy in enemies:
        dir_name += str(enemy) + "_"
    return dir_name[:-1]


def init_env(enemy, layer_nodes):
    """
    Initializes environment for single objective mode  with static enemy and ai player.
    """
    experiment_name = os.path.join(RUNS_DIR, 'enemy_' + enemies_dir_name(enemy))

    return Environment(
        experiment_name=experiment_name,
        playermode="ai",
        player_controller=PlayerController(layer_nodes),
        speed="fastest",
        enemymode="static",
        level=2,
        logs="off",
        savelogs="no",
        sound="off",
        randomini="yes"
    )


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


def main():
    # initialize the environment
    env = init_env(ENEMIES_TRAINING, LAYER_NODES[1])

    # dict containing all the results for each played games
    logbook = {}

    # iterate across the runs
    pattern = os.path.join(RUNS_DIR, 'enemy_' + enemies_dir_name(ENEMIES_TRAINING), BEST_INDIVIDUAL_PATTERN + "*[0-9].txt")
    for file in glob.glob(pattern):
        # extract the number of the run from the file name
        n_run = re.search("[0-9]+", os.path.basename(file)).group(0)
        print(f"RUN {n_run}:")

        # load the best individual
        best_individual_path = os.path.join(RUNS_DIR, 'enemy_' + enemies_dir_name(ENEMIES_TRAINING), BEST_INDIVIDUAL_PATTERN + n_run + ".txt")
        best_individual = np.loadtxt(best_individual_path)

        # play the game N_GAMES times
        individual_gains = [play_game(env, best_individual) for _ in range(N_GAMES)]
        [print(f"\tgame {game} - gain = {individual_gain}") for game, individual_gain in enumerate(individual_gains)]
        print()

        # save the results of the games of the current run
        logbook[n_run] = {"individual_gains":individual_gains}

    logbook_path = os.path.join(RUNS_DIR, "enemy_" + enemies_dir_name(ENEMIES_TRAINING), "games_played.csv")
    pd.DataFrame.from_dict(logbook, orient='index').to_csv(logbook_path, index=True, index_label='n_run', sep=";")
    print(
        f"Results of the games saved in {logbook_path} "
    )


if __name__ == "__main__":
    main()

