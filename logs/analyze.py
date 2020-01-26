#! /bin/python
import matplotlib.pyplot as plt
import numpy as np

from helpers import *

def main():
    all_runs = get_runs()
    lenet_runs = filter_runs(all_runs, model='lenet')

    kn = filter_runs(lenet_runs, init='kaiming-normal')
    ku = filter_runs(lenet_runs, init='kaiming-uniform')
    xn = filter_runs(lenet_runs, init='xavier-normal')
    xu = filter_runs(lenet_runs, init='xavier-uniform')
    normal = filter_runs(lenet_runs, init='normal')
    uniform = filter_runs(lenet_runs, init='uniform')
    xu_half = filter_runs(lenet_runs, init='xavier-uniform-half')
    xu_double = filter_runs(lenet_runs, init='xavier-uniform-double')
    xn_half = filter_runs(lenet_runs, init='xavier-normal-half')
    xn_double = filter_runs(lenet_runs, init='xavier-normal-double')

    # Set var to examine and gather run data in matrix
    var = 'test_acc'
    data = [np.array([run[var] for run in kn]),
            np.array([run[var] for run in ku]),
            np.array([run[var] for run in xn]),
            np.array([run[var] for run in xu]),
            np.array([run[var] for run in normal]),
            np.array([run[var] for run in uniform]),
            np.array([run[var] for run in xu_double]),
            np.array([run[var] for run in xu_half]),
            np.array([run[var] for run in xn_double]),
            np.array([run[var] for run in xn_half]),
            ]

    # Prepare line data to plot
    lines = []
    #lines.append(('kn', np.mean(data[0], axis=0), np.std(data[0], axis=0)))
    #lines.append(('ku', np.mean(data[1], axis=0), np.std(data[1], axis=0)))

    lines.append(('xu', np.mean(data[3], axis=0), np.std(data[3], axis=0)))
    lines.append(('xu_double', np.mean(data[6], axis=0), np.std(data[6], axis=0)))
    lines.append(('xu_half', np.mean(data[7], axis=0), np.std(data[7], axis=0)))

    #lines.append(('normal', np.mean(data[4], axis=0), np.std(data[4], axis=0)))
    #lines.append(('uniform', np.mean(data[5], axis=0), np.std(data[5], axis=0)))
    lines.append(('xn', np.mean(data[2], axis=0), np.std(data[2], axis=0)))
    lines.append(('xn_double', np.mean(data[8], axis=0), np.std(data[8], axis=0)))
    lines.append(('xn_half', np.mean(data[9], axis=0), np.std(data[9], axis=0)))

    pruning_rate = 0.2
    pruning_interval = 5
    start = 4
    total_epochs = 100
    X = range(int(total_epochs/pruning_interval))

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    xticklabels = 100 * (1-pruning_rate) ** np.arange(0,20)
    for ax in axs.flatten():
        ax.grid(True, linestyle=':')
        ax.set_xticks(range(20))
        ax.set_xticklabels(np.round(xticklabels, 1))
        ax.set_xlabel('% weights remaining')
        ax.set_ylabel(var)

    for i, line in enumerate(lines):
        y = line[1][start::pruning_interval]
        yerr = line[2][start::pruning_interval]
        axs.flatten()[i//3].errorbar(X, y, yerr=(yerr/2), label=line[0])

    for ax in axs.flatten():
        ax.legend()

    # Set other stuff
    plt.show()


if __name__ == "__main__":
    main()
