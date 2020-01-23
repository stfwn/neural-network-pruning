#! /bin/python
import matplotlib.pyplot as plt
import numpy as np

from helpers import *

def main():
    all_runs = get_runs()
    lenet_kn = filter_runs(all_runs, model='lenet', init='kaiming-normal')
    lenet_ku = filter_runs(all_runs, model='lenet', init='kaiming-uniform')
    lenet_xn = filter_runs(all_runs, model='lenet', init='xavier-normal')
    lenet_xu = filter_runs(all_runs, model='lenet', init='xavier-uniform')

    # Set var to examine and gather run data in matrix
    var = 'test_acc'
    data = [np.array([run[var] for run in lenet_kn]),
            np.array([run[var] for run in lenet_ku]),
            np.array([run[var] for run in lenet_xn]),
            np.array([run[var] for run in lenet_xu])]

    # Prepare line data to plot
    lines = []
    lines.append(('lenet_kn', np.mean(data[0], axis=0), np.std(data[0], axis=0)))
    lines.append(('lenet_ku', np.mean(data[1], axis=0), np.std(data[1], axis=0)))
    lines.append(('lenet_xn', np.mean(data[2], axis=0), np.std(data[2], axis=0)))
    lines.append(('lenet_xu', np.mean(data[3], axis=0), np.std(data[3], axis=0)))

    pruning_rate = 0.2
    pruning_interval = 5
    start = 4
    total_epochs = 100
    X = range(int(total_epochs/pruning_interval))

    fig, ax = plt.subplots(1, 1, constrained_layout=True)

    for line in lines:
        y = line[1][start::pruning_interval]
        yerr = line[2][start::pruning_interval]
        ax.errorbar(X, y, yerr=yerr, label=line[0])

    ax.set_xticks(range(20))
    # Set other stuff
    xticklabels = 100 * (1-pruning_rate) ** np.arange(0,20)
    ax.set_xticklabels(np.round(xticklabels, 1))
    ax.set_xlabel('% weights remaining')
    ax.set_ylabel(var)
    ax.legend()
    plt.show()
    # TODO: why are some errorbars > 100?







    

def plot_1():
    var = 'test_acc'
    a = np.mean(np.array([run[var] for run in lenet_kn]), axis=0)
    b = np.mean(np.array([run[var] for run in lenet_ku]), axis=0)
    c = np.mean(np.array([run[var] for run in lenet_xn]), axis=0)
    d = np.mean(np.array([run[var] for run in lenet_xu]), axis=0)
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.set_xlim(0,100)
    axs.set_ylim(50,100)
    axs.plot(a, label='lenet_kn')
    axs.plot(b, label='lenet_ku')
    axs.plot(c, label='lenet_xn')
    axs.plot(d, label='lenet_xu')

    axs.set_xlabel('epoch')
    axs.set_ylabel('test_acc')
    axs.legend()
    plt.show()


if __name__ == "__main__":
    main()
