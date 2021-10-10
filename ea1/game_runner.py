"""
This file contains the code for the game runner. 
What it does is creating an environment, running the game, and getting the final 
fitness score for the game.
"""
import datetime
import time
import sys

sys.path.insert(0, "evoman")
from environment import Environment
import os
from pathlib import Path
import numpy as np


# redefine multiple function for env
def custom_multiple(self, pcont, econt):
    vfitness, vplayerlife, venemylife, vtime = [], [], [], []
    for e in self.enemies:
        fitness, playerlife, enemylife, time = self.run_single(e, pcont, econt)
        vfitness.append(fitness)
        vplayerlife.append(playerlife)
        venemylife.append(enemylife)
        vtime.append(time)

    vfitness = self.cons_multi(np.array(vfitness))
    vplayerlife = sum(vplayerlife)
    venemylife = sum(venemylife)
    vtime = self.cons_multi(np.array(vtime))

    return vfitness, vplayerlife, venemylife, vtime


class GameRunner:
    def __init__(
        self,
        controller,
        enemies,
        experiment_name="",
        level=2,
        speed="fastest",
        headless=True,
    ):
        """
        This class instantiates an EVOMAN environment, runs the game and evaluates the fitness.
        """
        self.controller = controller
        self.enemies = enemies
        self.experiment_name = (
            experiment_name
            if experiment_name != ""
            else f"runs/logs/Run {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        )
        self.level = level
        self.speed = speed

        # redefine the method multiple for env
        Environment.multiple = custom_multiple
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.env = Environment(
                experiment_name=self.experiment_name,
                enemies=self.enemies,
                playermode="ai",
                multiplemode="yes",
                player_controller=self.controller,
                enemymode="static",
                level=self.level,
                speed=self.speed,
                logs="off",
                savelogs="no",
                sound="off",
                randomini="yes",
            )

        else:
            # Creates a directory for the experiment's logs
            Path(self.experiment_name).mkdir(parents=True, exist_ok=True)
            self.env = Environment(
                experiment_name=self.experiment_name,
                enemies=self.enemies,
                multiplemode="yes",
                playermode="ai",
                player_controller=self.controller,
                enemymode="static",
                level=self.level,
                speed=self.speed,
                randomini="yes",
            )

        self.env.state_to_log()

    def evaluate(self, individual):
        """
        Method to actually run a play simulation.
        :param individual: one individual from the population
        """

        fitness, player_life, enemy_life, time = self.env.play(pcont=individual["weights_and_biases"])
        return fitness, player_life, enemy_life, time


