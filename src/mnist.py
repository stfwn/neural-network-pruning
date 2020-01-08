#! /bin/python
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms

def main():
    print(f'Initializing MNIST datasets.')
    mnist_train = datasets.MNIST(root='../data/', train=True, download=True,
            transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='../data/', train=False, download=True,
            transform=transforms.ToTensor())
    print(f'Done.\n\t* len(mnist_train)={len(mnist_train)}\n\t* len(mnist_test)={len(mnist_test)}')

    train_loader = data.DataLoader(mnist_train, batch_size=64, shuffle=True)
    test_loader = data.DataLoader(mnist_test, batch_size=64, shuffle=True)

    model = MNISTFC()
    for batch_data, batch_targets in train_loader:
        print(model.forward(batch_data))


class MNISTFC(nn.Module):
    """ Fully connected (FC) neural network with defaults set to dimensions of
    the MNIST dataset, based on the example given by BrainCreators."""

    def __init__(self, device='cpu', hidden_dim=64, n_classes=10, in_features=28*28):
        super(MNISTFC, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=hidden_dim, out_features=n_classes, bias=True)
            ).to(device)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            raise TypeError

        # Flatten image to fit net input dimensions.
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)

if __name__ == "__main__":
    main()
