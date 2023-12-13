import argparse

from matplotlib import pyplot as plt
import pandas as pd
from hyper_parameter_optimization.result.optimization_result import OptimizationResult
import seaborn as sns
import numpy as np
sns.set()

from revolve2.ci_group import terrains
from revolve2.ci_group.simulation import make_standard_batch_parameters
from revolve2.experimentation.logging import setup_logging
from revolve2.experimentation.rng import make_rng_time_seed
from revolve2.modular_robot import ModularRobot
from revolve2.modular_robot.body import RightAngles
from revolve2.modular_robot.body.v1 import ActiveHingeV1, BodyV1, BrickV1
from revolve2.modular_robot.brain.cpg import BrainCpgNetworkNeighborRandom
from revolve2.modular_robot_simulation import ModularRobotScene, simulate_scenes
from revolve2.simulators.mujoco_simulator import LocalSimulator



def plot_mean_ci(a: np.ndarray, label='', title='', save_path=None, save_title=None):
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
    if save_path:
        plt.savefig(save_path+ '/' + save_title+'.png')



def simulate(individuals):

    # Create a modular robot scene.
    # This is a combination of one or more modular robots positioned in a given terrain.
    scene = ModularRobotScene(terrain=terrains.flat())


    for individual in individuals:
        body = individual.genotype.body.develop()
        brain = individual.genotype.brain.develop(body)
        robot = ModularRobot(body, brain)
        scene.add_robot(robot)

    # Create a simulator that will perform the simulation.
    # This tutorial chooses to use Mujoco, but your version of revolve might contain other simulators as well.
    simulator = LocalSimulator(num_simulators=1)

    # `batch_parameters` are important parameters for simulation.
    # Here, we use the parameters that are standard in CI Group.
    batch_parameters = make_standard_batch_parameters()

    # Simulate the scene.
    # A simulator can run multiple sets of scenes sequentially; it can be reused.
    # However, in this tutorial we only use it once.
    simulate_scenes(
        simulator=simulator,
        batch_parameters=batch_parameters,
        scenes=scene,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str)  #path to save result
    args = parser.parse_args()
    plot_save_path = '/'.join(args.name.split('/')[:-1])

    #Load result
    optimization_result = OptimizationResult.load(args.name)
    best_run = optimization_result.best_run()

    #Plot the utility over the runs
    optimization_result.plot_utility(use_timestamp=False)

    #Plot the best run max fitness over time
    best_run_max_fitness = best_run.get_stats('maxs')
    plot_mean_ci(best_run_max_fitness, label='Mean max fitness', title='Max fitness per generation for the best run', save_path=plot_save_path, save_title='1')

    #Plot the best run mean fitness over time
    best_run_mean_fitness = best_run.get_stats('means')
    plot_mean_ci(best_run_mean_fitness, label='Mean mean fitness', title='Mean fitness per generation for the best run', save_path=plot_save_path, save_title='2')

    #Plot the best run min fitness over time
    best_run_min_fitness = best_run.get_stats('mins')
    plot_mean_ci(best_run_min_fitness, label='Mean min fitness', title='Min fitness per generation for the best run', save_path=plot_save_path, save_title='3')

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

    plt.savefig(plot_save_path + '/' + 'correlations' +'.png')

    #Print the summary
    print('Summary table')
    print(optimization_result.summary())

    #Print the importance
    print('Parameter importance')
    print(pd.DataFrame([optimization_result.importance]))

    simulate([stats['winner_individual'] for stats in best_run.statistics])

    # sns.relplot(x="horsepower", y="mpg", hue="origin", size="weight",
    #         sizes=(40, 400), alpha=.5, palette="muted",
    #         height=6, data=mpg)
    






    plt.show()








