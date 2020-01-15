import os

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt


class Trainer():
    def __init__(self, model, dataset, device, batch_size=60, learning_rate=1.2e-3):
        if dataset == 'MNIST':
            self.dataset_name = dataset
            self.dataset = datasets.MNIST(root='./data/', train=True,
                    download=True, transform=transforms.ToTensor())
        else:
            raise ValueError(f'Dataset "{dataset}" not supported.')
        
        self.device = device
        self.model = model
        self.loader = data.DataLoader(self.dataset, batch_size=batch_size,
                shuffle=True)
        self.loss_function = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.losses = []
        self.accuracies = []

    def train_epoch(self):
        batch_losses = []

        for batch_data, batch_targets in self.loader:
            batch_data = batch_data.to(self.device)

            # Reset gradient to 0 (otherwise it accumulates)
            self.optimizer.zero_grad()

            predictions = self.model.forward(batch_data)
            loss = self.loss_function(predictions, batch_targets).to(self.device)

            # Compute delta terms and do a step of GD
            loss.backward()
            self.optimizer.step()

            # Keep track of loss
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)

        self.losses.append(epoch_loss)
        self.accuracies.append(self.__compute_accuracy())

    def __compute_accuracy(self):
        correct = 0
        for batch_data, batch_targets in self.loader:
            predictions = self.model.forward(batch_data)
            classifications = predictions.argmax(dim=-1,
                    keepdim=True).view_as(batch_targets)
            correct += classifications.eq(batch_targets).sum().item()
        accuracy = correct / len(self.dataset) * 100
        return accuracy

