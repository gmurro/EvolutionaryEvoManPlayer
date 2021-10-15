"""
Script to tune the hyperparameters.
Uses HyperOpt, a library for hyperparameter tuning.
The GeneticOptimizer object will take care of the evolution and just return the best individual's fitness.
"""
import os
import csv

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
from genetic_optimization import GeneticOptimizer
from game_runner import GameRunner
from singlelayer_controller import PlayerController
from hyperopt import hp, fmin, tpe
from hyperopt import SparkTrials

TUNING_DIR = "hyperparameter_tuning"

# fixed parameters
ENEMY = [2, 5, 8]
LAYER_NODES = [20, 10, 5]
GENERATIONS = 20
LAMBDA = 5
MUTATION_STEP_SIZE = 1.
N_ISLANDS = 10

# file name of the csv where store intermediate results
file_path = os.path.join(TUNING_DIR, "logbook_tuning.csv")


def write_header(file_path):
    """
    Write the header for the csv file
    """
    os.makedirs(TUNING_DIR, exist_ok=True)
    fields = ['pop_size', 'cx_prob', 'cx_alpha', 'mut_prob', 'mut_indpb', 'tournsize', 'migration_interval', 'migration_size', 'fitness_best',
              'gain_best']
    with open(file_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(fields)


def test_hyperparameters_vector(args):
    """
    Tests an hyperparameter vector with GeneticOptimizer
    """

    print(f"Now trying combination {args}")
    game_runner = GameRunner(PlayerController(LAYER_NODES[1]), enemies=ENEMY, headless=True)
    optimizer = GeneticOptimizer(
        enemies=ENEMY,
        layer_nodes=LAYER_NODES,
        n_islands=N_ISLANDS,
        population_size=int(args["population_size"]),
        game_runner=game_runner,
        generations=GENERATIONS,
        lambda_offspring=LAMBDA,
        cx_probability=round(args["cx_probability"], 2),
        cx_alpha=round(args["cx_alpha"], 2),
        mut_probability=round(args["mut_probability"], 2),
        mut_step_size=MUTATION_STEP_SIZE,
        mut_indpb=round(args["mutation_indpb"], 2),
        tournsize=int(args["tournsize"]),
        migration_interval=int(args["migration_interval"]),
        migration_size=int(args["migration_size"]),
        parallel=True,
    )
    best_individual = optimizer.evolve()

    # write the combination of hyperparameters used in this trial to this file
    row = [int(args['population_size']), round(args['cx_probability'], 2), round(args['cx_alpha'], 2),
           round(args['mut_probability'], 2), round(args['mutation_indpb'], 2), int(args['tournsize']),
           int(args['migration_interval']), int(args['migration_size']), best_individual['fitness'],
           best_individual['individual_gain']
           ]
    with open(file_path, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    return -best_individual["individual_gain"]


# initialize the headers in the csv file
write_header(file_path)

# define the search space across the hyperparameters
space = hp.choice(
    "GA",
    [
        {
            "population_size": hp.quniform("population_size", 6, 11, 1),
            "cx_probability": hp.uniform("cx_probability", 0.4, 1),
            "cx_alpha": hp.uniform("cx_alpha", 0, 1),
            "mut_probability": hp.uniform("mut_probability", 0, 0.6),
            "migration_interval": hp.quniform("migration_interval", 2, 10, 1),
            "migration_size": hp.quniform("migration_size", 2, 10, 1),
            "tournsize": hp.quniform("tournsize", 5, 9, 1),
            "mutation_indpb": hp.uniform("mutation_indpb", 0, 1),
        },
    ],
)
spark_trials = SparkTrials()
best = fmin(
    test_hyperparameters_vector,
    space,
    trials=spark_trials,
    algo=tpe.suggest,
    max_evals=72,
)

print("\nThe best combination of hyperparameters is:")
print(best)
