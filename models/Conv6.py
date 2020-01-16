import torch
import torch.nn as nn

from copy import deepcopy

class Conv6(nn.Module):
    """ Conv6 network, a varaint of VGG from Simonyan & Zisserman (2014), as
    used by Frankle & Carbin (2019) with in/out dimensions set to the
    dimensions of the MNIST dataset. """

    def __init__(self, device='cpu', in_features=28*28, out_features=10):
        super(Conv6, self).__init__()
        self.device = device
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64,
                    kernel_size=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64,
                    kernel_size=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=1),
                nn.Conv2d(in_channels=64, out_channels=128,
                    kernel_size=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128,
                    kernel_size=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=1),
                nn.Conv2d(in_channels=128, out_channels=256,
                    kernel_size=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256,
                    kernel_size=4),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=4, stride=1)).to(device)

        self.fc = nn.Sequential(
                nn.Linear(in_features=256, out_features=256, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=256, out_features=256, bias=True),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(in_features=256, out_features=out_features, bias=True)
            ).to(device)

        self.save_weights()

    def forward(self, x):
        if type(x) is not torch.Tensor:
            raise TypeError

        # Do convolutions
        x = self.conv(x).to(self.device)
        # Flatten image to fit fc input dimensions.
        x = x.view(x.shape[0], -1).to(self.device)
        return self.fc.forward(x)

    def save_weights(self):
        print('Saving weights.')
        # Deepcopy to avoid just saving references
        self.saved_weights = deepcopy(list(self.parameters()))

    def reset_weights(self):
        with torch.no_grad():
            for saved, current in zip(self.saved_weights, self.parameters()):
                current.data = saved.data
