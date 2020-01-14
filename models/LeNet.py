import torch
import torch.nn as nn

class LeNet(nn.Module):
    """ LeNet 300-100 network with in/out dimensions set to the dimensions of
    the MNIST dataset. """

    def __init__(self, device='cpu', in_features=28*28, out_features=10):
        super(LeNet, self).__init__()

        self.layers = nn.Sequential(
                nn.Linear(in_features=in_features, out_features=300, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=300, out_features=100, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=100, out_features=out_features, bias=True)
            ).to(device)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            raise TypeError

        # Flatten image to fit net input dimensions.
        x = x.view(x.shape[0], -1)
        return self.layers.forward(x)
