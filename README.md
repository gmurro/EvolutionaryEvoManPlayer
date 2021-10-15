# Assignment2
Second assignment, due 18/10/2021.

## Related Literature
- Parameter Tuning of Evolutionary Algorithms: Generalist vs. Specialist (Eiben) - https://www.few.vu.nl/~gusz/papers/2010-EVOSTAR-GeneralistSpecialist.pdf

## Tips for groups training
In the [multi_evolution](https://github.com/karinemiras/evoman_framework/blob/master/multi_evolution.pdf) paper are experimented trainings with 2 and 3 enemies:
- 2 enemies during the training
    - Enemies 2, 5 and 8 are frequently beaten even when they are not part of the training set. On
    the other hand, the enemies 4, 6 and 7 were only frequently
    beaten whenever they are part of the training set.
    - This means
    that the set (2, 5, 8) is robust to different strategies while the
    set (4, 6, 7) requires specific variations of a general strategy,
    thus implying they are more challenging enemies.
    - The enemies 1 and 3 were never beaten by any of the evolved
    strategies.
    - Best achieved result was obtained with the training set (7, 8) with a total of five victoriesù
    
- 3 enemies during the training
    - they generated a tendency of evolving an specific strategy to the training set
    - training set (1, 2, 5) achieved a generalization strategy for five enemies

## Notes for the competition
- Any changes you make to the provided network code, e.g., normalization changes, will be reflected on your version, but not in ours, and thus might harm the performance of your provided solution. Therefore, do not make any changes.

## How update fitness score?
To change the default implementation of `cons_multi` without editing the evoman folder, you should redefine a new function:
```python
def my_cons_multi(self,values):
 return values.mean()
```
and overwriting the old one:
```python
env.cons_multi = my_cons_multi
```

## Notes for executions
You will have two EAs implemented (EA1 and EA2). 
Also you will pick 2 groups of individuals of arbitrary size (group1 and group2). You will perform evolution on both groups, with both algorithms, so that is a total of 4 combinations (EA1 group1, EA1 group2, EA2 group1, EA2 group2). During evolution, you still record in each case the mean and max fitness. You will repeat each run 10 times. For each run, you will store the best performing individual of that experiment. When you have all your repetitions, you will create plots showing the mean of the mean and the mean of the max with the respective stds across the 10 runs. This is essentially the same as for assignment 1.

Now, however, you will take the best individuals, which will be 10 (runs) * 2 (EAs) * 2 (groups) = 40. Each one of them will now compete against ALL 8 ENEMIES (!) 5 times, independent of what groups they were trained on. You will record the individual gain values and average them across these 5 repetitions, resulting again in 40 values of mean individual gain. Then, you will plot these values in 4 boxplots where each box represents 1 EA - group combinations, so each box is made up of 10 individuals. These 4 sets of 10 values is also what you should do your statistical test(s) on. Interesting here is, if one EA can consistently come up with significantly better "best individuals" that the other EA.

When you run your experiments with 10 repetitions for a number of generations, for each repetition please record a history of the population mean fitness and maximum fitness for each generation. Then, as you have 10 of these records, please take the mean of the mean across all runs, the std of the means (NOT the mean of the stds), and the mean of the maximum values, also with std. This should result in 4 lines, each with a shaded area around them. A nice way to realize this kind of plot is by using seaborn (check this link (Links to an external site.) (Links to an external site.)). This requires you to have all the records in a pandas dataframe, which I can highly recommend using for this purpose.

Now, the best candidates, no matter what generation they showed up, should be tested against the respective enemy 5 times.  From the 5 results, you then make the average, so you again have 10 average performances of the best solution. These 10 values you should have 6 times (3 enemies, 2 EAs), and put them in 3 boxplots, one per enemy. If you ask me, you can also use a grouped boxplot. Here, I can again recommend to use seaborn, which makes the generation of the plot a piece of cake if your data is nicely arranged in a pandas dataframe (check this link (Links to an external site.) (Links to an external site.)).

### Experiments
Good tested enemies:
- [2, 5, 8] (tuned on these)
- [1,5,6]

### Tuning EA1

|      | pop_size | cx_prob | cx_alpha | mut_prob | mut_indpb | tournsize | migration_interval | migration_size | fitness_best | gain_best |
| ---: | -------: | ------: | -------: | -------: | --------: | --------: | -----------------: | -------------: | -----------: | --------: |
|    0 |        7 |     0.6 |      0.3 |     0.25 |       0.1 |         6 |                  3 |              7 |      89.7733 |     196.4 |
|    1 |        7 |    0.74 |      0.5 |     0.19 |      0.27 |         8 |                  5 |              8 |      89.5289 |     177.2 |
|    2 |        6 |    0.51 |     0.88 |     0.55 |      0.67 |         7 |                  7 |              7 |      85.8942 |     207.2 |
|    3 |        7 |     0.5 |     0.42 |     0.58 |      0.91 |         6 |                  3 |              8 |      82.8481 |     167.6 |
|    4 |        7 |    0.76 |     0.72 |     0.38 |      0.41 |         8 |                  8 |              5 |      87.1311 |     212.4 |
|    5 |        9 |    0.63 |     0.24 |     0.01 |      0.52 |         7 |                  6 |              4 |      86.9866 |       150 |
|    6 |        7 |    0.84 |     0.83 |     0.47 |      0.32 |         7 |                  5 |              9 |      88.1074 |     208.2 |
|    7 |        9 |    0.79 |     0.26 |      0.4 |      0.09 |         7 |                  4 |              2 |      90.1322 |       201 |
|    8 |       11 |    0.54 |     0.49 |     0.42 |      0.18 |         6 |                  7 |             10 |       87.241 |     186.8 |
|    9 |       10 |    0.62 |     0.31 |     0.54 |      0.99 |         7 |                  6 |              3 |      87.4822 |     214.6 |
|   10 |        7 |    0.58 |     0.04 |     0.24 |      0.95 |         6 |                  5 |              4 |      85.5189 |     205.8 |
|   11 |       10 |    0.77 |     0.26 |     0.57 |      0.96 |         6 |                  5 |              5 |       86.606 |     200.2 |
|   12 |        7 |    0.59 |     0.16 |     0.08 |      0.12 |         5 |                  9 |              8 |      89.4032 |     205.8 |
|   13 |       11 |    0.91 |     0.52 |     0.21 |      0.89 |         7 |                  7 |              4 |      83.8965 |     184.2 |
|   14 |       10 |    0.57 |     0.22 |     0.18 |       0.6 |         6 |                  6 |              6 |      89.3788 |     180.8 |
|   15 |        7 |    0.99 |     0.02 |     0.03 |      0.02 |         8 |                  2 |              3 |      89.5251 |       173 |
|   16 |        8 |    0.83 |     0.19 |     0.56 |      0.53 |         9 |                  9 |              6 |      87.3882 |     211.2 |
|   17 |        8 |    0.84 |     0.26 |     0.38 |      0.18 |         7 |                  4 |             10 |      89.2649 |     220.4 |
|   18 |        6 |    0.94 |     0.83 |     0.48 |      0.34 |         8 |                  8 |              9 |      89.7312 |     218.4 |
|   19 |       11 |    0.69 |     0.26 |     0.26 |      0.22 |         8 |                  9 |              4 |      87.6304 |     217.6 |
|   20 |        6 |    0.91 |     0.73 |     0.35 |      0.37 |         9 |                 10 |              5 |       84.585 |     178.8 |
|   21 |        8 |     0.4 |     0.63 |     0.45 |      0.42 |         9 |                  7 |              2 |      84.6988 |     164.6 |
|   22 |        9 |    0.42 |     0.99 |     0.34 |      0.78 |         8 |                  8 |              3 |      86.8694 |     186.2 |
|   23 |       11 |    0.84 |     0.42 |     0.02 |      0.95 |         9 |                  5 |              7 |      88.5927 |     154.4 |
|   24 |        6 |    0.73 |     0.73 |     0.28 |      0.85 |         7 |                  9 |              5 |      88.2726 |     191.2 |
|   25 |        8 |    0.45 |     0.59 |     0.31 |      0.66 |         5 |                  8 |              3 |      81.8532 |     191.4 |
|   26 |       10 |    0.66 |        1 |     0.14 |      0.42 |         8 |                 10 |              5 |      89.3378 |     206.4 |
|   27 |       10 |    0.67 |     0.67 |     0.49 |      0.76 |         8 |                  8 |              3 |      74.0911 |     199.6 |
|   28 |        9 |    0.74 |     0.36 |     0.53 |      0.76 |         8 |                  6 |              2 |      84.8095 |     180.8 |
|   29 |        6 |    0.97 |     0.88 |      0.6 |      0.17 |         8 |                  4 |              9 |      70.7718 |     179.8 |
|   30 |       10 |    0.63 |     0.11 |     0.44 |      0.44 |         9 |                  7 |              4 |      90.6167 |       225 |
|   31 |        6 |    0.96 |     0.56 |     0.42 |      0.27 |         7 |                  3 |             10 |      85.7177 |       236 |
|   32 |        8 |     0.9 |     0.09 |     0.52 |      0.28 |         7 |                  4 |              6 |      91.1866 |     243.6 |
|   33 |        6 |    0.78 |     0.95 |     0.38 |      0.36 |         9 |                  3 |             10 |      85.8309 |     170.8 |
|   34 |        7 |    0.81 |     0.36 |     0.21 |      0.24 |         7 |                 10 |              8 |      88.3477 |     201.6 |
|   35 |        9 |    0.88 |     0.79 |     0.42 |      0.22 |         8 |                  2 |              9 |      89.5995 |     194.8 |
|   36 |        8 |    0.94 |     0.45 |     0.36 |       0.3 |         5 |                  4 |              9 |      88.3099 |     199.2 |
|   37 |        8 |    0.93 |     0.13 |     0.49 |      0.01 |         6 |                  8 |              8 |      90.0773 |     225.6 |
|   38 |        6 |       1 |     0.47 |      0.4 |      0.33 |         6 |                  5 |              7 |       86.501 |     239.4 |
|   39 |        9 |    0.87 |     0.52 |     0.43 |      0.47 |         8 |                  2 |              9 |      90.0295 |     190.2 |
|   40 |        7 |    0.95 |     0.31 |     0.15 |       0.5 |         6 |                  6 |             10 |      87.1365 |     225.4 |
|   41 |       11 |     0.8 |     0.37 |     0.31 |      0.16 |         7 |                  7 |             10 |      88.5583 |     205.6 |
|   42 |        7 |     0.9 |     0.68 |     0.53 |      0.06 |         7 |                  3 |             10 |      87.9861 |     193.4 |
|   43 |        8 |    0.87 |     0.06 |     0.58 |      0.57 |         7 |                  3 |              8 |       88.361 |     230.6 |
|   44 |       11 |    0.54 |     0.07 |     0.45 |      0.46 |         5 |                  7 |              4 |      86.6136 |     202.2 |
|   45 |        6 |    0.71 |        0 |      0.6 |      0.05 |         6 |                  3 |             10 |      89.1758 |     180.2 |
|   46 |       10 |    0.63 |     0.56 |     0.52 |      0.27 |         9 |                  5 |              6 |      83.9974 |     218.2 |