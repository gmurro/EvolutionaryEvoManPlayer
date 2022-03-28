# 🧬 Evolutionary EvoMan player - Assignment 2 🎮

This repository contains a project realized as part of the *Evolutionary Computing* course of the Master's degree in Artificial Intelligence, Vrije Universiteit Amsterdam.     
The aim of this project is compare two **Evolutionary Algorithms** for the task of video game playing using a python framework called EvoMan.
The two EA methods train a *generalist agent* that should be able to play agains the different 8 enemies of the game.
For more details, read the [task assigment](standard_assignment_taskII.pdf).
The proposed solution is describer in the [report](report/report.pdf).


### Approach 1

For the first part of the experiment, meaning the evolution part, you need to:
- Run the `ea1/genetic_optimization.py` file. There you can set up the ENEMY global variable to change through different group of enemies. 
- The best individuals found are stored in the `ea1/runs/enemy_#/best_individual_run_#.txt` file. The history of the evolution is stored in the `ea1/runs/enemy_#/logbook_run_#.csv` file.
- The `ea1/hyperparameter_tuning.py` file is used to tune the hyperparameters of the algorithm through hyperopt. The history of the tuning is stored in the `ea1/hyperparameter_tuning/logbook_tuning.csv` file. 
- `ea1/experiment_runner.py` runs 10 optimizations simultaneously.
  
For the second part, where the best individuals are confronted to different enemies, you need to:
- Run the `ea1/play_with_best.py` file, selecting the enemy you want to play with. 
- Results are then stored in the `ea1/runs/enemy_#/games_played.csv` file.

### Approach 2

For the first part of the experiment, meaning the evolution part, you need to:
- Run the `ea2/genetic_optimization.py` file. There you can set up the ENEMY global variable to change through different group of enemies. 
- The best individuals found are stored in the `ea2/runs/enemy_#/best_individual_run_#.txt` file. The history of the evolution is stored in the `ea1/runs/enemy_#/logbook_run_#.csv` file.
- The `ea2/hyperparameter_tuning.py` file is used to tune the hyperparameters of the algorithm through hyperopt. 
- `ea21/experiment_runner.py` runs 10 optimizations simultaneously.
  
For the second part, where the best individuals are confronted to different enemies, you need to:
- Run the `ea2/play_with_best.py` file, selecting the enemy you want to play with. 
- Results are then stored in the `ea2/runs/enemy_#/games_played.csv` file.

### Experiments
Group of enemies trained on:
- [2, 5, 8] (tuned on these)
- [1, 5, 6]

### Results
The best individual found is `ea2/runs/enemy_1_5_6/best_individual_run_5.txt`, it is trained on the group of enemies [1, 5, 6] and it is able to defeat up to 5 enemies.
To have a deeper insight on the graphs, check out the `plots` folder.

![Run](./video.gif)

## Group members - 88

|  Name     |  Surname  |     Email                              |    Username      |
| :-------: | :-------: | :------------------------------------: | :--------------: |
| Simone  | Montali     | `s.montali@student.vu.nl`       | [_montali_](https://github.com/montali)         |
| Giuseppe  | Murro     | `g.murro@student.vu.nl`       | [_gmurro_](https://github.com/gmurro)         |
| Nedim | Azar | `n.azar@student.vu.nl` | [_nedimazar_](https://github.com/nedimazar) |
| Martin | Pucheu  Avilés    | `m.i.pucheuaviles@student.vu.nl`      | [_martinpucheuaviles_](https://github.com/martinpucheuaviles) |


## License

This project is licensed under the GNU General Public Licens - see the [LICENSE](LICENSE) file for details

