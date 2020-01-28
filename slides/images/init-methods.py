#! /bin/python
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform

def main():
    fan_in = 28*28
    fan_out = 300
    fan_mode = 'fan_in'

    mode = fan_in if fan_mode == 'fan_in' else fan_out
    xu = np.sqrt(6/(fan_in + fan_out))
    xn = np.sqrt(2/(fan_in + fan_out))
    ku = np.sqrt(3/mode)
    kn = 1 / np.sqrt(mode)

    x = np.linspace(start=norm.ppf(0.10), stop=norm.ppf(0.90), num=500)

    fig, axs = plt.subplots(2, 3, constrained_layout=True, figsize=(15,10))
    
    axs[0][0].plot(x, norm(0, xn).pdf(x), label='xavier-normal')
    axs[0][0].plot(x, norm(0, kn).pdf(x), label='kaiming-normal')
    axs[1][0].plot(x, uniform(-xu, 2*xu).pdf(x), label='xavier-uniform')
    axs[1][0].plot(x, uniform(-ku, 2*ku).pdf(x), label='kaiming-uniform')

    fan_in = 300
    fan_out = 100
    fan_mode = 'fan_in'

    mode = fan_in if fan_mode == 'fan_in' else fan_out
    xu = np.sqrt(6/(fan_in + fan_out))
    xn = np.sqrt(2/(fan_in + fan_out))
    ku = np.sqrt(3/mode)
    kn = 1 / np.sqrt(mode)

    axs[0][1].plot(x, norm(0, xn).pdf(x), label='xavier-normal')
    axs[0][1].plot(x, norm(0, kn).pdf(x), label='kaiming-normal')
    axs[1][1].plot(x, uniform(-xu, 2*xu).pdf(x), label='xavier-uniform')
    axs[1][1].plot(x, uniform(-ku, 2*ku).pdf(x), label='kaiming-uniform')


    fan_in = 100
    fan_out = 10
    fan_mode = 'fan_in'

    mode = fan_in if fan_mode == 'fan_in' else fan_out
    xu = np.sqrt(6/(fan_in + fan_out))
    xn = np.sqrt(2/(fan_in + fan_out))
    ku = np.sqrt(3/mode)
    kn = 1 / np.sqrt(mode)

    axs[0][2].plot(x, norm(0, xn).pdf(x), label='xavier-normal')
    axs[0][2].plot(x, norm(0, kn).pdf(x), label='kaiming-normal')
    axs[1][2].plot(x, uniform(-xu, 2*xu).pdf(x), label='xavier-uniform')
    axs[1][2].plot(x, uniform(-ku, 2*ku).pdf(x), label='kaiming-uniform')

    #fig.suptitle('Initialization Methods\' Probability Density Functions', fontsize=30)
    axs[0][0].set_title('Input Layer -> Hidden Layer 1')
    axs[0][1].set_title('Hidden Layer 1 -> Hidden Layer 2')
    axs[0][2].set_title('Hidden Layer 2 -> Ouput Layer')
    for ax in axs.flatten():
        ax.legend()

    plt.savefig('pdfs.png')

if __name__ == "__main__":
    main()
