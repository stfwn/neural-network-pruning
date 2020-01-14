import os

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt

def train(model, dataset='MNIST'):
    if dataset == 'MNIST':
        train_mnist(model)
    else:
        print('Unkown dataset.')
        raise ValueError


def train_mnist(model):
    print(f'Initializing MNIST training data.')
    mnist_train = datasets.MNIST(root='./data/', train=True, download=True,
            transform=transforms.ToTensor())
    train_loader = data.DataLoader(mnist_train, batch_size=60, shuffle=True)
    print(f'Initialization done.\n\t* len(mnist_train)={len(mnist_train)}')

    # Init loss function
    loss_function = nn.CrossEntropyLoss()

    # Connect optimizer to model params
    optimizer = optim.Adam(model.parameters(), lr=1.2e-3)

    losses = []

    """Frankle & Carbin (2019) do 50K iterations with batches of 60. The MNIST
    training set has 60k samples, so we should also do the same:
    (50k * 60) / 60000 = 50 epochs."""
    # RANGE SHOULD BE 50
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

    # Save model
    now = dt.now().strftime('%Y-%m-%d-%H-%M')
    os.makedirs('./models/states/', exist_ok=True)
    torch.save(model.state_dict(), f'./models/states/{now}.pt')

    plt.plot(losses)
    plt.show()
