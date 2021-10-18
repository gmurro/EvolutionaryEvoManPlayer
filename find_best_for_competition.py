import os
import pandas as pd
import numpy as np
from ast import literal_eval
import sys
from ea1.singlelayer_controller import PlayerController

sys.path.insert(0, "evoman")
from environment import Environment

N_GAMES = 5
LAYER_NODES = [20, 10, 5]
GAMES_FILE_NAME = "games_played.csv"
BEST_INDIVIDUAL_PATTERN = "best_individual_run_"

# enemies to play against
ENEMIES_TEST = [1, 2, 3, 4, 5, 6, 7, 8]


def find_best(df, value1, value2):
    """ Find the maximal value in the df sorting first according to value1 and then according to value2.
    Return the index of the best. """
    occurrences_max_value1 = np.where(
        df[value1] == df[value1].max()
    )[0]

    return occurrences_max_value1[
            np.argmax(df.iloc[occurrences_max_value1][value2])
        ]


def play_game(env, best_individual):
    """
    Play a game against all enemies and return the tables of the results
    """

    games_df = pd.DataFrame({
        "enemy" : pd.Series(dtype='str'),
        "player_life" : pd.Series(dtype='float'),
        "enemy_life": pd.Series(dtype='float'),
        "games_won" : pd.Series(dtype='int')
    })

    stats = {}

    for i, enemy in enumerate(ENEMIES_TEST):

        player_life_list = np.zeros(N_GAMES)
        enemy_life_list = np.zeros(N_GAMES)
        fitness_list = np.zeros(N_GAMES)
        games_won = 0

        # Update the enemy
        env.update_parameter('enemies', [enemy])

        print(f"Enemy {enemy}:")
        for game in range(N_GAMES):
            fitness, player_life, enemy_life, time = env.play(best_individual)

            player_life_list[game] = player_life
            enemy_life_list[game] = enemy_life
            fitness_list[game] = fitness
            if player_life > enemy_life:
                 games_won += 1
        print(f"\tavg fitness: {fitness_list.mean()}")
        print(f"\tavg individual gain: {np.array([p - e for (p, e) in zip(player_life_list,enemy_life_list )]).mean()}")
        print(f"\tgames won: {games_won}/{N_GAMES}")

        stats[enemy] = {"fitness" : fitness_list,
                        "player_life" : player_life_list,
                        "enemy_life": enemy_life_list
                        }
        games_df.loc[i] = [enemy, round(player_life_list.mean(), 1), round(enemy_life_list.mean(), 1), int(games_won/N_GAMES * 100)]

    final_gain = np.zeros(N_GAMES)
    final_fitness = np.zeros(N_GAMES)
    for game in range(N_GAMES):
        sum_player_life = 0
        sum_enemy_life = 0
        fitness_multi = []
        for enemy in ENEMIES_TEST:
            sum_player_life += stats[enemy]["player_life"][game]
            sum_enemy_life += stats[enemy]["enemy_life"][game]
            fitness_multi.append(stats[enemy]["fitness"][game])

        fitness_multi = np.array(fitness_multi)
        final_gain[game] = sum_player_life - sum_enemy_life
        final_fitness[game] = fitness_multi.mean() - fitness_multi.std()

    print(f"Avg gain measure: {final_gain.mean()}")
    print(f"Avg fitness measure: {final_fitness.mean()}\n")

    # return the dataframe of the games against each enemy
    return games_df


# read all games files
games_files = []
for dirpath, dirnames, filenames in os.walk("."):
    for filename in [f for f in filenames if f == GAMES_FILE_NAME]:
        games_files.append(os.path.join(dirpath, filename))

# dataframe of best individuals for each EA and for each group of enemies trained on
all_bests = pd.DataFrame({'controller': pd.Series(dtype='str'),
                          'individual_gains': pd.Series(dtype='float'),
                          'defeated_enemies': pd.Series(dtype='float')
                          })

for i, file in enumerate(games_files):
    df_games = pd.read_csv(file, sep=";")
    # convert the element inside the column 'individual_gains' and 'defeated_enemies' to an array
    df_games['individual_gains'] = df_games['individual_gains'].apply(literal_eval)
    df_games['defeated_enemies'] = df_games['defeated_enemies'].apply(literal_eval)

    # compute the mean of the individual_gains and defeated_enemies for each run
    df_games['individual_gains'] = df_games['individual_gains'].map(lambda x: np.array(x).mean())
    df_games['defeated_enemies'] = df_games['defeated_enemies'].map(lambda x: np.array(x, dtype=np.float32).mean())

    # find max of the means of the defeated_enemies
    id_max = find_best(df_games, 'defeated_enemies', 'individual_gains')

    best_run = int(df_games.iloc[id_max]["n_run"])
    best_mean_gain = df_games.iloc[id_max]["individual_gains"]
    best_mean_defeated_enemies = df_games.iloc[id_max]["defeated_enemies"]

    # store the n run of the best and its gain
    controller_best = os.path.join(os.path.dirname(file), BEST_INDIVIDUAL_PATTERN+ str(best_run)+".txt")
    all_bests.loc[i] = [controller_best, best_mean_gain, best_mean_defeated_enemies]


# find the best individual across all EAs
id_best = find_best(all_bests, 'defeated_enemies', 'individual_gains')
best_for_competition = all_bests.iloc[id_best]
print(f"The best individual found is {best_for_competition['controller']} "
      f"(gain = {round(best_for_competition['individual_gains'],2)}, "
      f"defeated_enemies = {round(best_for_competition['defeated_enemies'],2)})")

# initialize the environment
env = Environment(
        experiment_name=os.path.dirname(best_for_competition['controller']),
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
best_individual = np.loadtxt(best_for_competition['controller'])
df = play_game(env, best_individual)
print(df.T.to_latex(header=False))
