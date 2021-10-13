"""
Script to tune the hyperparameters.
Uses HyperOpt, a library for hyperparameter tuning.
The GeneticOptimizer object will take care of the evolution and just return the best individual's fitness.
"""
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
os.environ["SDL_VIDEODRIVER"] = "dummy"
from genetic_optimization import GeneticOptimizer
from game_runner import GameRunner
from singlelayer_controller import PlayerController
from hyperopt import hp, fmin, tpe
from hyperopt import SparkTrials

# fixed parameters
ENEMY = [2, 5, 8]
LAYER_NODES = [20, 10, 5]
GENERATIONS = 20
LAMBDA = 5
MUTATION_STEP_SIZE = 1.


def test_hyperparameters_vector(args):
    """
    Tests an hyperparameter vector with GeneticOptimizer
    """
    print(f"Now trying combination {args}")
    game_runner = GameRunner(PlayerController(LAYER_NODES), enemies=ENEMY, headless=True)
    optimizer = GeneticOptimizer(
        enemies=ENEMY,
        layer_nodes=LAYER_NODES,
        population_size=int(args["population_size"]),
        game_runner=game_runner,
        generations=GENERATIONS,
        lambda_offspring = LAMBDA,
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
    print(f"TRIAL with:")
    print(f"pop_size={int(args['population_size'])}, cx_prob={round(args['cx_probability'], 2)}, cx_alpha={round(args['cx_alpha'], 2)}, "
          f"mut_prob={round(args['mut_probability'], 2)}, mut_indpb={round(args['mutation_indpb'], 2)}, tournsize={int(args['tournsize'])},"
          f"migration_interval={int(args['migration_interval'])}, migration_size={int(args['migration_size'])}")
    print(f"fitness_best={best_individual['fitness']}")
    print(f"gain_best={best_individual['individual_gain']}")
    return -best_individual["fitness"]


space = hp.choice(
    "GA",
    [
        {
            "population_size": hp.quniform("population_size", 5, 10, 1),
            "cx_probability": hp.uniform("cx_probability", 0.4, 1),
            "cx_alpha": hp.uniform("cx_alpha", 0, 1),
            "mut_probability": hp.uniform("mut_probability", 0, 0.6),
            "migration_interval": hp.quniform("migration_interval", 2, 10, 1),
            "migration_size": hp.quniform("migration_size", 2, 10, 1),
            "tournsize": hp.quniform("tournsize", 5, 10, 1),
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
