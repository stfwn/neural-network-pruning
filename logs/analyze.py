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
    

    var = 'test_acc'
    a = np.mean(np.array([run[var] for run in lenet_kn]), axis=0)
    b = np.mean(np.array([run[var] for run in lenet_ku]), axis=0)
    c = np.mean(np.array([run[var] for run in lenet_xn]), axis=0)
    d = np.mean(np.array([run[var] for run in lenet_xu]), axis=0)
    fig, axs = plt.subplots(1, 1, constrained_layout=True)
    axs.plot(a, label='lenet_kn')
    axs.plot(b, label='lenet_ku')
    axs.plot(c, label='lenet_xn')
    axs.plot(d, label='lenet_xu')
    axs.set_xlabel('epochs')
    axs.set_ylabel('test_acc')
    axs.legend()
    plt.show()


if __name__ == "__main__":
    main()
