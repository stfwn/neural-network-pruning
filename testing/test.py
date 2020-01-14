from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

def test(model, dataset='MNIST'):
    if dataset == 'MNIST':
        test_mnist(model)
    else:
        raise ValueError(f'Dataset "{dataset}" not supported.')

def test_mnist(model):
    print(f'Initializing MNIST testing data.')
    mnist_test = datasets.MNIST(root='./data/', train=False, download=True,
            transform=transforms.ToTensor())
    test_loader = data.DataLoader(mnist_test, batch_size=60, shuffle=True)
    print(f'Initialization done.\n\t* len(mnist_test)={len(mnist_test)}')

    correct = 0
    for batch_data, batch_targets in test_loader:
        predictions = model.forward(batch_data)
        classifications = predictions.argmax(dim=-1, keepdim=True).view_as(batch_targets)
        correct += classifications.eq(batch_targets).sum().item()
    print(f'Accuracy on test set: {correct / len(mnist_test) * 100}%')
