#! /bin/python
import torch
from torchvision import datasets, transforms

def main():
    print(f'Initializing MNIST datasets.')
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True,
            transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='../data/', train=False, download=True,
            transform=transforms.ToTensor())
    print(f'Done!\n\t* len(mnist_train)={len(mnist_train)}\n\t* len(mnist_test)={len(mnist_test)}')

if __name__ == "__main__":
    main()
