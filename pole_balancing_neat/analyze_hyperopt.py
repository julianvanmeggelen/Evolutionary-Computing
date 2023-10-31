from hyperopt import get_dir

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set()
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
def plot_runs(a, name='', use_std=True, color=None):
    #a = np.stack(a)
    mean= np.mean(a,axis=0)
    max= np.max(a,axis=0)
    print(mean[-1])
    min= np.min(a,axis=0)
    std = np.std(a,axis=0)
    if not use_std:
        plt.fill_between(x=range(a.shape[1]), y1 = min, y2 = max,alpha=0.5, label='', color=color)
        plt.plot(mean, label= f"Avg. of {name} fitness", color=color)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()
    else:
        plt.fill_between(x=range(a.shape[1]), y1 = mean-std, y2 = mean+std, alpha=0.5, label='')
        plt.plot(mean, label= f"Avg. of {name} fitness")
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend()


def plot_net_sizes(net_size_nodes, net_size_connections, all_fitness, name):
    net_size_nodes_merged = np.array(net_size_nodes).flatten()
    net_size_connections_merged = np.array(net_size_connections).flatten()
    all_fitness = np.array(all_fitness).flatten()
    print(f"{all_fitness.shape=}")
    print(f"{net_size_nodes_merged.shape=}")


    plt.figure()
    plt.hist(net_size_nodes_merged)#[-1])
    plt.title(f'#nodes per network, {name} (all populations merged)')
    plt.figure()
    plt.hist(net_size_nodes[-1])
    plt.title(f'#nodes per network, {name} (One randomly selected population)')

    # Nr connections
    plt.figure()
    plt.hist(net_size_connections_merged)
    plt.title(f'#connections per network, {name}  (all populations merged)')

    plt.figure()
    plt.hist(net_size_connections[-1])
    plt.title(f'#connections per network, {name}  (One randomly selected population)')

    #nodes vs fitness

    plt.figure()
    plt.title(f'#nodes per network vs. fitness, {name}  (All populations merged)')
    sns.regplot(x=net_size_nodes_merged, y=all_fitness)
    plt.xlabel('#nodes')
    plt.ylabel('fitness')


    plt.figure()
    plt.title(f'#connections per network vs. fitness, {name}  (All populations merged)')
    sns.regplot(x=net_size_connections_merged, y=all_fitness)
    plt.xlabel('#connections')
    plt.ylabel('fitness')

    plt.figure()
    plt.title(f'#connections per node vs. fitness, {name}  (All populations merged)')
    sns.regplot(x=net_size_connections_merged/net_size_nodes_merged, y=all_fitness)
    plt.xlabel('#connections/node')
    plt.ylabel('fitness')

    plt.figure()
    plt.title(f'#nodes per network vs. fitness, {name}  (All populations merged)')
    sns.boxplot( x=net_size_nodes_merged, y=np.log(all_fitness))
    plt.xlabel('#nodes')
    plt.ylabel('fitness (logarithmic)')

    plt.figure()
    plt.title(f'#connections per network vs. fitness, {name}  (All populations merged)')
    sns.boxplot(x=net_size_connections_merged, y=np.log(all_fitness))
    plt.xlabel('#connections')
    plt.ylabel('fitness (logarithmic)')

    


if __name__ == "__main__":
    study_name = os.environ.get('STUDY', None)
    if study_name is None:
        study_name= 'test935dfb8e-915d-428f-a969-105d002d95d2'

    import optuna

    study = optuna.create_study(study_name=study_name, storage='sqlite:///hyperopt_results.sqlite', load_if_exists=True)
    print(len(study.trials))
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(df)

    #Plot the best
    best_id = study.best_trial._trial_id
    trial_dir = get_dir(best_id, study_name)

    mins = np.load(f'{trial_dir}/maxs.npy')
    means = np.load(f'{trial_dir}/maxs.npy')
    maxs = np.load(f'{trial_dir}/mins.npy')
    print(mins[0])

    plot_runs(maxs, 'max')
    plot_runs(means, 'mean', color='orange')

    # with open(f'{trial_dir}/config.pkl', 'rb') as f:
    #     best_config = pickle.load(f)

    #plot the current
    current_dir = './trial_results/current'
    mins_current = np.load(f'{current_dir}/mins.npy')
    means_current = np.load(f'{current_dir}/means.npy')
    maxs_current = np.load(f'{current_dir}/maxs.npy')
    plt.figure()
    plot_runs(maxs_current, 'max')
    plot_runs(means_current, 'mean', color='orange')
    plt.figure()
    plot_runs(maxs_current, 'max')
    plt.figure()
    plot_runs(means_current, 'mean', color='red')
    plt.figure()

    #plot hists
    print(maxs.shape)
    sns.kdeplot(maxs[:,-1], label='Tuned')
    sns.kdeplot(maxs_current[:,-1], label='Original')
    plt.xlim(0,70)
    plt.xlabel('Fitness')
    plt.legend()

    #Plot net sizes, original
    net_size_nodes = np.load(f'{current_dir}/net_size_nodes.npy')
    net_size_connections = np.load(f'{current_dir}/net_size_connections.npy')
    all_fitness = np.load(f'{current_dir}/all_fitness.npy')

    plot_net_sizes(net_size_nodes, net_size_connections, all_fitness, name='original')

    #Plot net sizes, tuned
    net_size_nodes = np.load(f'{trial_dir}/net_size_nodes.npy')
    net_size_connections = np.load(f'{trial_dir}/net_size_connections.npy')
    all_fitness = np.load(f'{trial_dir}/all_fitness.npy')

    plot_net_sizes(net_size_nodes, net_size_connections, all_fitness, name='tuned')
    plt.show()
   






