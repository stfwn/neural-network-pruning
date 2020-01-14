#! /bin/python
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

def main():
    print(f'Initializing MNIST datasets.')
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True,
            transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='../data/', train=False, download=True,
            transform=transforms.ToTensor())
    print(f'Done.\n\t* len(mnist_train)={len(mnist_train)}\n\t* len(mnist_test)={len(mnist_test)}')

    train_loader = data.DataLoader(mnist_train, batch_size=60, shuffle=True)
    test_loader = data.DataLoader(mnist_test, batch_size=60, shuffle=True)

    # Init model and loss function
    model = MNISTFC()
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


class MNISTFC(nn.Module):
    """ LeNet 300-100 network with in/out dimensions set to the dimensions of
    the MNIST dataset. """

    def __init__(self, device='cpu'):
        super(MNISTFC, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(in_features=28*28, out_features=300, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=300, out_features=100, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=100, out_features=10, bias=True)
            ).to(device)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            raise TypeError

        # Flatten image to fit net input dimensions.
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)

if __name__ == "__main__":
    main()
