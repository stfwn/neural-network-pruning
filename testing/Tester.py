from torch.utils import data
import torch.nn as nn
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

class Tester():
    def __init__(self, model, dataset, device, batch_size=60):
        if dataset == 'MNIST':
            self.dataset_name = dataset
            self.dataset = datasets.MNIST(root='./data/', train=False,
                    download=True, transform=transforms.ToTensor())
        elif dataset == 'CIFAR10':
            self.dataset_name = dataset
            self.dataset = datasets.CIFAR10(root='./data/', train=False,
                    download=True, transform=transforms.ToTensor())
        else:
            raise ValueError(f'Dataset "{dataset}" not supported.')

        self.device = device
        self.model = model
        self.loader = data.DataLoader(self.dataset, batch_size=batch_size,
                shuffle=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.losses = []
        self.accuracies = []

    def test_epoch(self):
        correct = 0
        batch_losses = []
        for batch_data, batch_targets in self.loader:
            batch_data = batch_data.to(self.device)
            batch_targets = batch_targets.to(self.device)

            predictions = self.model.forward(batch_data)
            classifications = predictions.argmax(dim=-1,
                    keepdim=True).view_as(batch_targets)
            correct += classifications.eq(batch_targets).sum().item()

            loss = self.loss_function(predictions, batch_targets)
            batch_losses.append(loss.item())
        self.losses.append(np.mean(batch_losses))
        self.accuracies.append(correct / len(self.dataset) * 100)
