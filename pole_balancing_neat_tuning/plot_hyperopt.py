import argparse

from matplotlib import pyplot as plt
import pandas as pd
from hyper_parameter_optimization.result.optimization_result import OptimizationResult
import seaborn as sns
import numpy as np
sns.set()



def plot_mean_ci(a: np.ndarray, label='', title=''):
    plt.figure()
    mean = np.mean(a,axis=0)
    std = np.std(a,axis=0)
    x = range(len(mean))
    plt.fill_between(x, y1=mean-std, y2=mean+std, alpha=0.4)
    plt.plot(mean, label=f'{label}($\pm 1 std$)')
    plt.xlabel('Generation')
    plt.ylabel('fitness')
    plt.legend()
    plt.title(title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)  #path to save result
    args = parser.parse_args()

    #Load result
    optimization_result = OptimizationResult.load(args.name)
    best_run = optimization_result.best_run()

    #Plot the utility over the runs
    optimization_result.plot_utility(use_timestamp=False)

    #Plot the best run max fitness over time
    best_run_max_fitness = best_run.get_stats('maxs')
    plot_mean_ci(best_run_max_fitness, label='Mean max fitness', title='Max fitness per generation for the best run')

    #Plot the best run mean fitness over time
    best_run_mean_fitness = best_run.get_stats('means')
    plot_mean_ci(best_run_mean_fitness, label='Mean mean fitness', title='Mean fitness per generation for the best run')

    #Plot the best run min fitness over time
    best_run_min_fitness = best_run.get_stats('mins')
    plot_mean_ci(best_run_min_fitness, label='Mean min fitness', title='Min fitness per generation for the best run')

    #plot scatterplot for each parameter vs utility
    n_params = len(optimization_result.tune_params)
    ncols = 4
    nrows = int(n_params/ncols)
    fig, axs = plt.subplots(nrows, ncols)
    fig.suptitle('Parameter value vs. utility relation')
    for param, ax in zip(optimization_result.tune_params.keys() ,axs.flatten()):
        param_vals = [getattr(run.config, param) for run in optimization_result.runs]
        utility_vals = [run.utility for run in optimization_result.runs]
        sns.scatterplot(x=param_vals, y=utility_vals, ax=ax)#, lowess=True)
        ax.set_xlabel(param)
        ax.set_ylabel('Utility')

    #Print the summary
    print('Summary table')
    print(optimization_result.summary())

    #Print the importance
    print('Parameter importance')
    print(pd.DataFrame([optimization_result.importance]))




    # sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
    #         sizes=(40, 400), alpha=.5, palette="muted",
    #         height=6, data=mpg)
    






    plt.show()








