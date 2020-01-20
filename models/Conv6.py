import torch
import torch.nn as nn
from copy import deepcopy
from models.ExpandedModule import ExpandedModule

class Conv6(ExpandedModule):
    """ Conv6 network, a varaint of VGG from Simonyan & Zisserman (2014), as
    used by Frankle & Carbin (2019) with in/out dimensions set to the
    dimensions of the CIFAR-10 dataset. """

    def __init__(self, device='cpu'):
        super(Conv6, self).__init__()
        self.device = device
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),

                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)).to(device)

        self.fc = nn.Sequential(
                nn.Linear(in_features=4096, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=256),
                nn.ReLU(),
                nn.Linear(in_features=256, out_features=10)
            ).to(device)

        self.save_weights()

    def forward(self, x):
        if type(x) is not torch.Tensor:
            raise TypeError

        # Do convolutions
        x = self.conv(x).to(self.device)
        # Flatten image to fit fc input dimensions.
        x = x.flatten(start_dim=1)
        return self.fc.forward(x)
