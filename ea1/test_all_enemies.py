#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                                 		  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        general solution for enemies (games)                                         #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

sys.path.insert(0, 'evoman')
from environment import Environment
from singlelayer_controller import PlayerController

# imports other libs
import numpy as np

experiment_name = 'controller_generalist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Update the number of neurons for this specific example
n_hidden_neurons = 10

# initializes environment for multi objetive mode (generalist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
                  playermode="ai",
                  player_controller=PlayerController(n_hidden_neurons),
                  speed="fastest",
                  enemymode="static",
                  level=2,
                  randomini='yes')

sol = np.loadtxt('runs/enemy_[1, 5, 7, 8]/best_individual_run_11.txt')
print('\n LOADING SAVED GENERALIST SOLUTION FOR ALL ENEMIES \n')

# tests saved demo solutions for each enemy
for en in range(1, 9):
    # Update the enemy
    env.update_parameter('enemies', [en])

    env.play(sol)

print('\n  \n')