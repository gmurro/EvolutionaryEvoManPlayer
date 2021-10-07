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

## Notes for executions
You will have two EAs implemented (EA1 and EA2). 
Also you will pick 2 groups of individuals of arbitrary size (group1 and group2). You will perform evolution on both groups, with both algorithms, so that is a total of 4 combinations (EA1 group1, EA1 group2, EA2 group1, EA2 group2). During evolution, you still record in each case the mean and max fitness. You will repeat each run 10 times. For each run, you will store the best performing individual of that experiment. When you have all your repetitions, you will create plots showing the mean of the mean and the mean of the max with the respective stds across the 10 runs. This is essentially the same as for assignment 1.

Now, however, you will take the best individuals, which will be 10 (runs) * 2 (EAs) * 2 (groups) = 40. Each one of them will now compete against ALL 8 ENEMIES (!) 5 times, independent of what groups they were trained on. You will record the individual gain values and average them across these 5 repetitions, resulting again in 40 values of mean individual gain. Then, you will plot these values in 4 boxplots where each box represents 1 EA - group combinations, so each box is made up of 10 individuals. These 4 sets of 10 values is also what you should do your statistical test(s) on. Interesting here is, if one EA can consistently come up with significantly better "best individuals" that the other EA.

When you run your experiments with 10 repetitions for a number of generations, for each repetition please record a history of the population mean fitness and maximum fitness for each generation. Then, as you have 10 of these records, please take the mean of the mean across all runs, the std of the means (NOT the mean of the stds), and the mean of the maximum values, also with std. This should result in 4 lines, each with a shaded area around them. A nice way to realize this kind of plot is by using seaborn (check this link (Links to an external site.) (Links to an external site.)). This requires you to have all the records in a pandas dataframe, which I can highly recommend using for this purpose.

Now, the best candidates, no matter what generation they showed up, should be tested against the respective enemy 5 times.  From the 5 results, you then make the average, so you again have 10 average performances of the best solution. These 10 values you should have 6 times (3 enemies, 2 EAs), and put them in 3 boxplots, one per enemy. If you ask me, you can also use a grouped boxplot. Here, I can again recommend to use seaborn, which makes the generation of the plot a piece of cake if your data is nicely arranged in a pandas dataframe (check this link (Links to an external site.) (Links to an external site.)).
