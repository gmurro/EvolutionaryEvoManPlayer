import time
import numpy as np
import random
from game_runner import GameRunner
import pickle
import os
from tqdm import tqdm
from singlelayer_controller import PlayerController
from scipy.spatial import distance_matrix
from tabulate import tabulate
import pandas as pd

N_RUN = 11
ENEMY = [1, 5, 6]
RUNS_DIR = "runs"
N_ISLANDS = 10  # Note that with 1 island, this is disabled
MIGRATION_INTERVAL = 4  # Please let this be higher than 1
MIGRATION_SIZE = 6

# We can now fix the number of nodes to be used in our NN. The first HAS TO BE the number of inputs.
# The last HAS TO BE the number of outputs. The middle HAS TO BE the number of nodes in the hidden layer.
LAYER_NODES = [20, 10, 5]

LAMBDA_REGULARIZATION = 0.01

# Then, we can instantiate the Genetic Hyperparameters.
CX_PROBABILITY = 0.9
CX_ALPHA = 0.56
MUT_PROBABILITY = 0.52
MUT_MU = 0
MUT_STEP_SIZE = 1.0
MUT_INDPB = 0.28
POPULATION_SIZE = 8  # Individuals PER EACH ISLAND
GENERATIONS = 30
SAVING_FREQUENCY = 3
TOURNSIZE = 7
LAMBDA = 5  # literature advise to use LAMBDA=5-7
MIN_VALUE_INDIVIDUAL = -1
MAX_VALUE_INDIVIDUAL = 1
EPSILON_UNCORRELATED_MUTATION = 1.0e-6
ALPHA_FITNESS_SHARING = 1.0
# [K. Deb. Multi-objective Optimization using Evolutionary Algorithms. Wiley, Chichester, UK, 2001]
# suggests that a default value for the niche size should be in the range 5–10
# set it to 0.0 to disable the fitness sharing algorithm
# If you are using the island model, DO NOT SET THIS over 0.
NICHE_SIZE = 0.0

def enemies_dir_name(enemies):
    """
    Converts a list of enemies into a string
    """
    dir_name = ""
    for enemy in enemies:
        dir_name += str(enemy) + "_"
    return dir_name[:-1]



class GeneticOptimizer:
    def __init__(
        self,
        game_runner,
        enemies,
        generations=GENERATIONS,
        layer_nodes=LAYER_NODES,
        cx_probability=CX_PROBABILITY,
        cx_alpha=CX_ALPHA,
        tournsize=TOURNSIZE,
        mut_probability=MUT_PROBABILITY,
        population_size=POPULATION_SIZE,
        lambda_offspring=LAMBDA,
        mut_mu=MUT_MU,
        mut_step_size=MUT_STEP_SIZE,
        mut_indpb=MUT_INDPB,
        niche_size=NICHE_SIZE,
        run_number=N_RUN,
        migration_interval=MIGRATION_INTERVAL,
        saving_frequency=SAVING_FREQUENCY,
        n_islands=N_ISLANDS,
        migration_size=MIGRATION_SIZE,
        checkpoint="checkpoint",
        parallel=False,
    ):
        """
        Initializes the Genetic Optimizer.
            :param layer_nodes: The number of nodes in each layer. (list)
            :param enemies: List of the enemies to defeat. (list)
            :param generations: The number of generations to run the GA for. (int)
            :param cx_probability: The probability of crossover. (float, 0<=x<=1)
            :param cx_alpha: Parameter of the crossover. Extent of the interval in which the new values can be drawn
                for each attribute on both side of the parents’ attributes. (float)
            :param tournsize: The size for the tournment in  the selection. (int)
            :param mut_probability: The probability of mutation. (float, 0<=x<=1)ù
            :param lambda_offspring: The scaling factor of the offspring size based on the population size
            :param mut_mu: The mean of the normal distribution used for mutation. (float)
            :param mut_step_size: The initial standard deviation of the normal distribution used for mutation. (float)
            :param mut_indpb: The probability of an individual being mutated. (float, 0<=x<=1)
            :param population_size: The size of the population. (int)
            :param niche_size: The size of the niche considered to keep diversity with the fitness sharing.
                                If it is 0.0, the fitness sharing will be disabled. (float)
            :param checkpoint: The file name to save the checkpoint. (str)
            :param game_runner: The EVOMAN game runner. (GameRunner)
        """
        self.layer_nodes = layer_nodes
        self.checkpoint = checkpoint
        self.enemies = enemies
        # The biases have to be the same amount of the nodes without considering the first layer
        self.bias_no = np.sum(self.layer_nodes) - self.layer_nodes[0]
        self.generations = generations
        self.cx_probability = cx_probability
        self.mut_probability = mut_probability
        self.population_size = population_size
        self.lambda_offspring = lambda_offspring
        self.game_runner = game_runner
        self.parallel = parallel
        self.niche_size = niche_size
        self.mut_mu = mut_mu
        self.tournsize = tournsize
        self.mut_step_size = mut_step_size
        self.mut_indpb = mut_indpb
        self.cx_alpha = cx_alpha
        self.logbook = {}
        self.run_number = run_number
        self.migration_interval = migration_interval
        self.saving_frequency = saving_frequency
        self.n_islands = n_islands
        self.migration_size = migration_size

        weights_no = 0
        for i in range(0, len(self.layer_nodes) - 1):
            weights_no += self.layer_nodes[i] * self.layer_nodes[i + 1]
        self.individual_size = weights_no + self.bias_no

        # compute the learning rate as suggested by the book
        # it is usually inversely proportional to the square root of the problem size
        self.learning_rate = 1 / (self.individual_size ** 0.5)
        self.verify_checkpoint()
        self.register_variation_operators()

    def getLogbook(self):
        return self.logbook

    def create_population(self, n, wandb_size):
        """
        Creates a population of n individuals.
        Each individual consists in a np.array containing the weights of the NN, and an array containing the mutation step size.
            :param n: The size of the population. (int)
            :param wandb_size: The size of the array containing weights and biases. (int)
        """
        population = [[] for i in range(self.n_islands)]
        for j, island in enumerate(population):
            for i in range(n):
                individual = {
                    "weights_and_biases": np.random.uniform(
                        low=MIN_VALUE_INDIVIDUAL,
                        high=MAX_VALUE_INDIVIDUAL,
                        size=wandb_size,
                    ),
                    "mut_step_size": self.mut_step_size,
                    "fitness": None,
                    "individual_gain": None,
                }
                population[j].append(individual)
        return population

    def intermediate_crossover(self, parent1, parent2):
        child1 = {
            "weights_and_biases": np.zeros(len(parent1["weights_and_biases"])),
            "mut_step_size": parent1["mut_step_size"],
            "fitness": None,
            "individual_gain": None,
        }
        child2 = {
            "weights_and_biases": np.zeros(len(parent2["weights_and_biases"])),
            "mut_step_size": parent1["mut_step_size"],
            "fitness": None,
            "individual_gain": None,
        }

        for i, (xi, yi) in enumerate(
            zip(parent1["weights_and_biases"], parent2["weights_and_biases"])
        ):
            w = np.random.rand(1)[0]

            if xi < yi:
                child1["weights_and_biases"][i] = xi + w * (yi - xi)
                child2["weights_and_biases"][i] = yi
            else:
                child2["weights_and_biases"][i] = xi + w * (yi - xi)
                child1["weights_and_biases"][i] = yi

        return child1, child2

    def blend_crossover(self, individual1, individual2):
        """
        Mates two individuals, randomly choosing a crossover point and performing a blend crossover as seen in book at page 67.
            :param individual1: The first parent. (np.array)
            :param individual2: The second parent. (np.array)
            :param alpha: Extent of the interval in which the new values can be drawn for each attribute on both side
                of the parents’ attributes. (float)
        """
        # For each weight/bias in the array, we decide a random shift quantity
        assert len(individual1["weights_and_biases"]) == len(
            individual2["weights_and_biases"]
        )
        for i in range(len(individual1["weights_and_biases"])):
            crossover = (1 - 2 * self.cx_alpha) * random.random() - self.cx_alpha
            individual1["weights_and_biases"][i] = (
                crossover * individual1["weights_and_biases"][i]
                + (1 - crossover) * individual2["weights_and_biases"][i]
            )
            # Then, we invert the two "crossover weights" for the second individual
            individual2["weights_and_biases"][i] = (
                crossover * individual2["weights_and_biases"][i]
                + (1 - crossover) * individual1["weights_and_biases"][i]
            )
        # We can then mutate the sigmas too!
        crossover = (1 - 2 * self.cx_alpha) * random.random() - self.cx_alpha
        individual1["mut_step_size"] = (
            crossover * individual1["mut_step_size"]
            + (1 - crossover) * individual2["mut_step_size"]
        )
        individual2["mut_step_size"] = (
            crossover * individual2["mut_step_size"]
            + (1 - crossover) * individual1["mut_step_size"]
        )
        return individual1, individual2

    def mutate_individual(self, individual):
        """
        Mutates an individual using a Gaussian distribution.
            :param individual: The individual to mutate. (np.array)
            :param indpb: The probability of a single weight of the individual being mutated. (float, 0<=x<=1)
        """
        # We mutate the weights and biases
        for i in range(len(individual["weights_and_biases"])):
            if random.random() < self.mut_indpb:
                individual["weights_and_biases"][i] += (
                    random.gauss(0, 1) * individual["mut_step_size"]
                )
        return individual

    def tournament_selection(self, population, k, tournsize):
        """
        Selects the best individuals from a population.
            :param population: The population to select from. (list)
            :param k: The number of individuals to select. (int)
            :param tournsize: The number of individuals participating in each tournament. (int)
        """
        chosen_ones = []
        for i in range(0, k):
            tournament = random.choices(population, k=tournsize)
            chosen_ones.append(max(tournament, key=lambda x: x["fitness"]))
        return chosen_ones

    def best_selection(self, population, k):
        """
        Selects the best individuals from a population.
            :param population: The population to select from. (list)
            :param k: The number of individuals to select. (int)
        """
        return sorted(population, key=lambda x: x["fitness"])[-k:]

    def verify_checkpoint(self):
        """
        Tries to load the checkpoint if it exists, otherwise it creates the population.
        """

        # Create the checkpoint directory  if it does not exist
        if not self.parallel:
            os.makedirs(
                os.path.join(RUNS_DIR, "enemy_" + enemies_dir_name(self.enemies), self.checkpoint),
                exist_ok=True,
            )

        # We have to define also an evaluation to compute the fitness sharing, if it is enabled
        if self.niche_size > 0:
            if not self.parallel:
                print(
                    f"Evolutionary process started using the 'Fitness sharing' method with niche_size={self.niche_size}"
                )

        checkpoint_path = os.path.join(
            RUNS_DIR,
            "enemy_" + enemies_dir_name(self.enemies),
            self.checkpoint,
            self.checkpoint + "_run_" + str(self.run_number) + ".dat",
        )
        if (not self.parallel) and os.path.isfile(checkpoint_path):
            # If the checkpoint file exists, load it.
            with open(checkpoint_path, "rb") as cp_file:
                cp = pickle.load(cp_file)
                self.population = cp["population"]
                if self.population_size == len(self.population):
                    self.start_gen = cp["generation"]
                    self.logbook = cp["logbook"]
                    random.setstate(cp["rndstate"])
                    print(
                        f"We got a checkpoint and the sizes coincide! Starting from generation no. {self.start_gen}"
                    )
                else:
                    print(
                        "We got a checkpoint, but it was for a different population size. Gotta start from scratch."
                    )
                    self.initialize_population()
        else:
            self.initialize_population()

    def sharing(self, distance, niche_size, alpha):
        """
        Sharing function which count distant neighbourhoods less than close neighbourhoods
            :param distance: Distance between two individuals (float)
            :param niche_size: It is the share radius, the size of a niche in the genotype space; it decides both how
                many niches can be maintained and the granularity with which different niches are discriminated (float)
            :param alpha: Determines the shape of the sharing function; for α=1 the function is linear, but for values
                greater than this the effect of similar individuals in reducing a solution’s fitness falls off more
                rapidly with distance (float)
        """
        if distance < niche_size:
            return 1 - (distance / niche_size) ** alpha
        else:
            return 0

    def fitness_sharing(self, individual, population, niche_size, alpha):
        """
        Compute the fitness of an individual and adjust it according to the number of individuals falling within some
        pre-specified distance.
            :param individual: individual for which you want compute the fitness sharing.
                It must have the individual["fitness"] != None. (dict)
            :param population: array of individual inside the actual population
            :param niche_size: It is the share radius, the size of a niche in the genotype space; it decides both how
                many niches can be maintained and the granularity with which different niches are discriminated (float)
            :param alpha: Determines the shape of the sharing function; for α=1 the function is linear, but for values
                greater than this the effect of similar individuals in reducing a solution’s fitness falls off more
                rapidly with distance (float)
        """
        # compute the fitness of the individual
        fitness = individual["fitness"]

        # compute array of distances between the individual and all other individual in the population
        distances = distance_matrix(
            [individual["weights_and_biases"]],
            [individual["weights_and_biases"] for individual in population],
        )[0]

        return fitness / sum([self.sharing(d, niche_size, alpha) for d in distances])

    def uncorrelated_mutation_one_step_size(
        self, mut_step_size, mu, learning_rate, epsilon
    ):
        """
        Update of the mutation step size. It must be computed before of performing the mutation on the individual.
        """
        mut_step_size *= np.exp(random.gauss(mu, learning_rate))

        # if the new mut_step_size is too small, return epsilon
        return mut_step_size if mut_step_size > epsilon else epsilon

    def initialize_population(self):
        self.population = self.create_population(
            self.population_size, self.individual_size
        )
        self.start_gen = 0
        self.logbook = {}

    def register_variation_operators(self):
        """
        Register the variation operators to be used in the evolution.
        """
        self.mutation_operators = [self.mutate_individual]
        self.crossover_operators = [self.blend_crossover, self.intermediate_crossover]
        self.variation_operators = [
            {
                "mutation": random.choice(self.mutation_operators),
                "crossover": random.choice(self.crossover_operators),
            }
            for i in range(self.n_islands)
        ]

    def evaluate_fitness_for_individuals(self, population):
        """
        This loops over a given population of individuals,
        and saves the fitness to each Individual object (individual.fitness.values)
        :param population: The population of individuals to evaluate. (list)
        """
        fitnesses = map(self.game_runner.evaluate, population)
        for ind, (fit, player_life, enemy_life, time) in zip(population, fitnesses):
            # compute regularization term l2
            weights_slice = LAYER_NODES[0] * LAYER_NODES[1] + LAYER_NODES[1]
            weights = np.concatenate(
                [
                    ind["weights_and_biases"][LAYER_NODES[1] : weights_slice],
                    ind["weights_and_biases"][weights_slice + LAYER_NODES[2] :],
                ]
            )

            ind["fitness"] = (
                fit - sum([w ** 2 for w in weights]) * LAMBDA_REGULARIZATION
            )
            ind["individual_gain"] = player_life - enemy_life

    def compute_fitness_sharing_for_individuals(self, population):
        """
        This loops over a given population of individuals, and computes the fitness to each individual
            :param population: The population of individuals to evaluate. It is required that for each individual the
            fitness value should be already computed (list)
        """
        fitnesses_sharing = map(
            lambda individual: self.fitness_sharing(
                individual,
                population=population,
                niche_size=self.niche_size,
                alpha=ALPHA_FITNESS_SHARING,
            ),
            population,
        )
        for ind, fit in zip(population, fitnesses_sharing):
            ind["fitness"] = fit

    def clone_individual(self, individual):
        """
        Clone an individual.
        """

        return {
            "weights_and_biases": individual["weights_and_biases"].copy(),
            "mut_step_size": individual["mut_step_size"],
            "fitness": individual["fitness"],
            "individual_gain": individual["individual_gain"],
        }

    def evaluate_stats(self, population):
        """
        Compute the statistics of the population.
            :param population: The population of individuals to evaluate. (list)
        """

        return {
            "avg_fitness": np.average(
                [ind["fitness"] for island in population for ind in island]
            ),
            "min_fitness": np.min(
                [ind["fitness"] for island in population for ind in island]
            ),
            "max_fitness": np.max(
                [ind["fitness"] for island in population for ind in island]
            ),
            "std_fitness": np.std(
                [ind["fitness"] for island in population for ind in island]
            ),
            "avg_mut_step_size": np.average(
                [ind["mut_step_size"] for island in population for ind in island]
            ),
            "std_mut_step_size": np.std(
                [ind["mut_step_size"] for island in population for ind in island]
            ),
            "islands_avg_fitness": np.array(
                [
                    np.average([ind["fitness"] for ind in island])
                    for island in population
                ]
            ),
            "best_individual": self.clone_individual(
                max(
                    [ind for island in population for ind in island],
                    key=lambda ind: ind["individual_gain"],
                )
            ),
        }

    def print_stats(self):
        """
        Pretty-prints the evolution stats
        """
        data = [
            [
                i,
                generation["avg_fitness"],
                generation["max_fitness"],
                generation["std_fitness"],
                generation["best_individual"]["fitness"],
                generation["best_individual"]["individual_gain"],
                generation["best_individual"]["mut_step_size"],
                generation["avg_mut_step_size"],
                generation["std_mut_step_size"],
            ]
            for i, generation in self.logbook.items()
        ]
        print(
            tabulate(
                data,
                headers=[
                    "Generation",
                    "Fitness avg",
                    "Fitness max",
                    "Fitness std",
                    "Fitness best",
                    "Gain best",
                    "mut_step_size best",
                    "mut_step_size avg",
                    "mut_step_size std",
                ],
                tablefmt="orgtbl",
            )
        )

    def find_best(self):
        """
        Find the individual with the best gain across all the generations. If there are more than one individuals with
        the same gain, return the individual with the highest fitness.
        """
        best_gain_along_generations = np.array(
            [
                self.logbook[i]["best_individual"]["individual_gain"]
                for i in range(self.start_gen + self.generations)
            ]
        )

        occurrences_max_best = np.where(
            best_gain_along_generations == best_gain_along_generations.max()
        )[0]

        return self.logbook[
            occurrences_max_best[
                np.argmax(
                    [
                        self.logbook[i]["best_individual"]["fitness"]
                        for i in occurrences_max_best
                    ]
                )
            ]
        ]["best_individual"]

    def migration(self, migration_size, generation):
        """
        Migrates some individuals (results show that it's better than just migrating the best)
        from an island to the next one.
        How do we select the individuals to migrate? We merge the approaches seen in:
        - https://www.sciencedirect.com/science/article/pii/S0305054815002361
        - https://link.springer.com/article/10.1007/s12065-019-00253-2 (DDMP)

        We take the worst individuals, and migrate them to an attractive island (having high fitness avg. increase)
        We then take individuals from that island, and send them to the aforementioned one
        """
        if len(self.population) < 2:  # If there is only one island, we don't migrate
            return
        for island_i, island in enumerate(self.population):
            # get the worst individuals
            worst_individuals_indices = sorted(
                range(0, len(self.population[island_i])),
                key=lambda i: self.population[island_i][i]["fitness"],
                reverse=True,
            )[:migration_size]

            # We pick an interesting island, basing on the fitness average increase
            avg_increases = (
                self.logbook[generation - 1]["islands_avg_fitness"]
                - self.logbook[generation - 2]["islands_avg_fitness"]
            )

            for individual_i in worst_individuals_indices:
                interesting_island = random.choices(
                    range(self.n_islands), weights=avg_increases, k=1
                )[0]
                # Pick a random individual in interesting_island to exchange this with
                interesting_island_individual = random.randint(
                    0, len(self.population[interesting_island]) - 1
                )
                # Exchange the individuals
                (
                    self.population[interesting_island][interesting_island_individual],
                    self.population[island_i][individual_i],
                ) = (
                    self.population[island_i][individual_i],
                    self.population[interesting_island][interesting_island_individual],
                )

    def evolve(self):
        """
        Runs the GA for a given number of generations.
        """

        # First, evaluate the whole population's fitnesses
        for i, island in enumerate(self.population):
            self.evaluate_fitness_for_individuals(island)

        # store the initial statistics about population 0 in the logbook
        self.logbook[self.start_gen] = self.evaluate_stats(self.population)

        # start the evolution across the generations
        for g in tqdm(
            range(self.start_gen + 1, self.start_gen + self.generations),
            desc=f"Run with nodes: {self.layer_nodes}",
            leave=False,
        ):
            if not self.parallel:
                print(
                    f"\n👨‍👩‍👧 Generation {g-1} is about to give birth to children! 👨‍👩‍👧"
                )

            # if the fitness sharing is enabled, you have to compute it for every island's population
            if self.niche_size > 0:
                for i, island in enumerate(self.population):
                    self.compute_fitness_sharing_for_individuals(island)
            if self.migration_size > 0 and g % self.migration_interval == 0:
                self.migration(self.migration_size, g)
            # create a new offspring of size LAMBDA*len(population)
            offspring_size = self.lambda_offspring * len(self.population[0])
            offspring = [[] for i in range(self.n_islands)]
            for island_i, island in enumerate(self.population):
                for i in range(1, offspring_size, 2):
                    # selection of 2 parents with replacement
                    parents = self.tournament_selection(
                        self.population[island_i], k=2, tournsize=self.tournsize
                    )

                    # clone the 2 parents in the new offspring
                    offspring[island_i].append(self.clone_individual(parents[0]))
                    offspring[island_i].append(self.clone_individual(parents[1]))

                    # apply mutation between the parents in a non-deterministic way
                    if random.random() < self.cx_probability:
                        (
                            offspring[island_i][i - 1],
                            offspring[island_i][i],
                        ) = self.variation_operators[island_i]["crossover"](
                            offspring[island_i][i - 1],
                            offspring[island_i][i],
                        )
                        offspring[island_i][i - 1]["fitness"] = None
                        offspring[island_i][i]["fitness"] = None

                    # apply mutation to the 2 new children
                    if random.random() < self.mut_probability:
                        # mutate the step size
                        offspring[island_i][i - 1][
                            "mut_step_size"
                        ] = self.uncorrelated_mutation_one_step_size(
                            offspring[island_i][i - 1]["mut_step_size"],
                            mu=self.mut_mu,
                            learning_rate=self.learning_rate,
                            epsilon=EPSILON_UNCORRELATED_MUTATION,
                        )
                        offspring[island_i][i][
                            "mut_step_size"
                        ] = self.uncorrelated_mutation_one_step_size(
                            offspring[island_i][i]["mut_step_size"],
                            mu=self.mut_mu,
                            learning_rate=self.learning_rate,
                            epsilon=EPSILON_UNCORRELATED_MUTATION,
                        )

                        # mutate the individuals
                        offspring[island_i][i - 1] = self.variation_operators[island_i][
                            "mutation"
                        ](offspring[island_i][i - 1])
                        offspring[island_i][i - 1]["fitness"] = None

                        offspring[island_i][i] = self.variation_operators[island_i][
                            "mutation"
                        ](offspring[island_i][i])
                        offspring[island_i][i]["fitness"] = None

            start_time = time.time()

            if self.niche_size > 0:
                # Evaluate the fitness for the whole offspring
                for i, island in enumerate(offspring):
                    self.evaluate_fitness_for_individuals(island)
            else:
                # If the fitness sharing is disabled, is not needed to recalculate the fitness each individual

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [
                    ind
                    for island_offspring in offspring
                    for ind in island_offspring
                    if ind["fitness"] is None
                ]

                # Then evaluate the fitness of individuals with an invalid fitness
                self.evaluate_fitness_for_individuals(invalid_ind)

            if not self.parallel:
                print(
                    f"Time to evaluate the fitness in the offspring: {round(time.time() - start_time, 3)} seconds"
                )

            # Select the survivors for next generation of individuals only between the new generation
            # (age-based selection)
            for island_i, island in enumerate(self.population):
                offspring[island_i] = self.best_selection(
                    offspring[island_i], len(self.population[island_i])
                )

            # The population is entirely replaced by the offspring
            self.population = offspring

            # We save every SAVING_FREQUENCY generations.
            if g % SAVING_FREQUENCY == 0 and not self.parallel:
                # Fill the dictionary using the dict(key=value[, ...]) constructor
                cp = dict(
                    population=self.population,
                    generation=g,
                    logbook=self.logbook,
                    rndstate=random.getstate(),
                )

                checkpoint_path = os.path.join(
                    RUNS_DIR,
                    "enemy_" + enemies_dir_name(self.enemies),
                    self.checkpoint,
                    self.checkpoint + "_run_" + str(self.run_number) + ".dat",
                )
                with open(checkpoint_path, "wb") as cp_file:
                    pickle.dump(cp, cp_file)

            # Compute the stats for the generation, and save them to the logbook.
            self.record = self.evaluate_stats(self.population)
            self.logbook[g] = self.record
            if not self.parallel:
                self.print_stats()

        # Return the best individual across all generations
        return self.find_best()


def run_optimization(run_number, enemies):
    """
    Runs the experiment
    """
    game_runner = GameRunner(
        PlayerController(LAYER_NODES[1]), enemies=enemies, headless=True
    )
    optimizer = GeneticOptimizer(
        population_size=POPULATION_SIZE,
        enemies=enemies,
        generations=GENERATIONS,
        game_runner=game_runner,
        run_number=run_number,
    )
    best_individual = optimizer.evolve()
    if not optimizer.parallel:
        # save the best individual in the best_individual.txt file
        best_individual_path = os.path.join(
            RUNS_DIR,
            "enemy_" + enemies_dir_name(enemies),
            "best_individual_run_" + str(run_number) + ".txt",
        )
        np.savetxt(best_individual_path, best_individual["weights_and_biases"])
        print(
            f"Evolution is finished! I saved the best individual in {best_individual_path} "
            f"(fitness={best_individual['fitness']}, gain={best_individual['individual_gain']})"
        )

        # save the logbook in a csv file
        logbook = optimizer.getLogbook()
        logbook_path = os.path.join(
            RUNS_DIR,
            "enemy_" + enemies_dir_name(enemies),
            "logbook_run_" + str(run_number) + ".csv",
        )
        pd.DataFrame.from_dict(logbook, orient="index").to_csv(
            logbook_path, index=True, index_label="n_gen", sep=";"
        )


if __name__ == "__main__":
    run_optimization(N_RUN, ENEMY)
