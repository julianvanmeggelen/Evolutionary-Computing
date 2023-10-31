import main
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sns.set()
N_RUNS =  20

if __name__ == "__main__":

    mins, maxs, means, sizes_nodes, sizes_connections, all_fitnesses = [],[],[],[],[],[]
    for i in range(N_RUNS):
        _, mins_curr, maxs_curr, means_curr, net_size_nodes, net_size_connections, all_fitness = main.main(plot=False)
        mins.append(mins_curr)
        maxs.append(mins_curr)
        means.append(means_curr)
        sizes_connections.append(net_size_connections)
        sizes_nodes.append(net_size_nodes)
        all_fitnesses.append(all_fitness)

    maxs = np.stack(maxs)
    print(maxs.shape)
    mean= np.mean(maxs,axis=0)
    max= np.max(maxs,axis=0)
    min= np.min(maxs,axis=0)
    std = np.std(maxs,axis=0)

    np.save(f'./trial_results/current/mins.npy', np.stack(mins))
    np.save(f'./trial_results/current/means.npy', np.stack(means))
    np.save(f'./trial_results/current/maxs.npy', np.stack(maxs))
    np.save(f'./trial_results/current/net_size_nodes.npy', sizes_nodes)
    np.save(f'./trial_results/current/net_size_connections.npy', sizes_connections)
    np.save(f'./trial_results/current/all_fitness.npy', all_fitnesses)

    plt.fill_between(x=range(maxs.shape[1]), y1 = min, y2 = max,alpha=0.5, label='')
    plt.plot(mean, label= "Avg. of max. fitness")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    plt.figure()
    plt.fill_between(x=range(maxs.shape[1]), y1 = mean-std, y2 = mean+std,alpha=0.5, label='')
    plt.plot(mean, label= "Avg. of max. fitness ($\pm 1 std$)")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    means = np.stack(means)
    print(means.shape)
    mean= np.mean(means,axis=0)
    max= np.max(means,axis=0)
    min= np.min(means,axis=0)
    std = np.std(means,axis=0)
    plt.figure()
    plt.fill_between(x=range(means.shape[1]),color='red', y1 = min, y2 = max,alpha=0.5)
    plt.plot(mean, color='red', label= "Avg. of avg. fitness")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    plt.figure()
    plt.fill_between(x=range(maxs.shape[1]),color='red', y1 = mean-std, y2 = mean+std,alpha=0.5, label='')
    plt.plot(mean, color='red', label= "Avg. of avg. fitness ($\pm 1 std$)")
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()

    plt.figure()
    plt.show()