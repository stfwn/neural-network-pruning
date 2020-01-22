#! /bin/python
import matplotlib.pyplot as plt
import numpy as np

import helpers

def main():
    all_runs = helpers.get_runs()

    lenet_kaiming_runs = []
    lenet_xavier_runs = []
    for run in all_runs:
        if run['args']['model'] == 'lenet' and 'kaiming' in run['args']['initialization']:
            lenet_kaiming_runs.append(run)
        else:
            lenet_xavier_runs.append(run)

    a = lenet_xavier_runs[0]
    b = lenet_xavier_runs[1]
    c = lenet_kaiming_runs[0]
    d = lenet_kaiming_runs[1]
    var = 'test_acc'

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    axs[0][0].plot(a[var], label=a['args']['initialization'])
    axs[0][0].plot(b[var], label=b['args']['initialization'])
    axs[0][0].plot(c[var], label=c['args']['initialization'])
    axs[0][0].plot(d[var], label=d['args']['initialization'])
    axs[0][0].set_xlabel('epoch')
    axs[0][0].set_ylabel(var)
    axs[0][0].legend()
    plot_run_diff(axs[0][1], a, b, var)
    plot_run_diff(axs[1][0], a, c, var)
    plot_run_diff(axs[1][1], a, d, var)
    plt.show()

def plot_run_diff(ax, a, b, var):
    diff = abs(a[var] - b[var])
    ax.set_xlabel('epoch')
    ax.set_ylabel(var)
    ax.plot(diff, label=f"difference between {a['args']['initialization']} and {b['args']['initialization']}")
    ax.legend()




if __name__ == "__main__":
    main()
