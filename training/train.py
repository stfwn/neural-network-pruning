import torch.optim as optim
import torch.nn as nn

from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

def train(model, dataset='MNIST'):
    if dataset == 'MNIST':
        train_mnist(model)
    else:
        print('Unkown dataset.')
        raise ValueError


def train_mnist(model):
    print(f'Initializing MNIST datasets.')
    mnist_train = datasets.MNIST(root='./data/', train=True, download=True,
            transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='./data/', train=False, download=True,
            transform=transforms.ToTensor())
    print(f'Done.\n\t* len(mnist_train)={len(mnist_train)}\n\t* len(mnist_test)={len(mnist_test)}')

    train_loader = data.DataLoader(mnist_train, batch_size=60, shuffle=True)
    test_loader = data.DataLoader(mnist_test, batch_size=60, shuffle=True)

    # Init loss function
    loss_function = nn.CrossEntropyLoss()

    # Connect optimizer to model params
    optimizer = optim.Adam(model.parameters(), lr=1.2e-3)

    losses = []

    """Frankle & Carbin (2019) do 50K iterations with batches of 60. The MNIST
    training set has 60k samples, so we should also do the same:
    (50k * 60) / 60000 = 50 epochs."""
    for epoch in range(50):
        print(f'Training epoch {epoch}.')
        epoch_loss = []
        for batch_data, batch_targets in train_loader:

            # Reset gradient to 0 (otherwise it accumulates)
            optimizer.zero_grad()

            predictions = model.forward(batch_data)
            loss = loss_function(predictions, batch_targets)

            # Compute delta terms and do a step of GD
            loss.backward()
            optimizer.step()

            # Keep track of loss
            epoch_loss.append(loss.item())
        losses.append(np.mean(epoch_loss))

    plt.plot(losses)
    plt.show()

    # Evaluate
    correct = 0
    for batch_data, batch_targets in test_loader:
        predictions = model.forward(batch_data)
        classifications = predictions.argmax(dim=-1, keepdim=True).view_as(batch_targets)
        correct += classifications.eq(batch_targets).sum().item()
    print(f'Accuracy on test set: {correct / len(mnist_test) * 100}%')
