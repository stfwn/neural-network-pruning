import os

import torch
import torch.optim as optim
import torch.nn as nn

from torch.utils import data
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from pruning import *

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

        # Instantiate the mask
        self.mask = init_mask(model)

        self.loader = data.DataLoader(self.dataset, batch_size=batch_size,
                shuffle=True)
        self.loss_function = nn.CrossEntropyLoss() 
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.losses = []
        self.accuracies = []

    def train_epoch(self, epoch_num):
        batch_losses = []

        # Every n epochs, prune
        # I hardcoded some values, (prune 20% every 5 epochs),
        # but you should add it as a
        # parameter in argparse, as well as the rate
        if epoch_num % 5 == 0 and epoch_num>0:
            update_mask(self.model, self.mask, 0.2)
        
        for batch_data, batch_targets in self.loader:
            batch_data = batch_data.to(self.device)
            batch_targets = batch_targets.to(self.device)

            # Reset gradient to 0 (otherwise it accumulates)
            self.optimizer.zero_grad()

            predictions = self.model.forward(batch_data)
            loss = self.loss_function(predictions, batch_targets).to(self.device)

            # Compute delta terms and do a step of GD
            loss.backward()
            self.optimizer.step()
            apply_mask(self.model, self.mask)
            # Keep track of loss
            batch_losses.append(loss.item())
        epoch_loss = np.mean(batch_losses)

        self.losses.append(epoch_loss)
        self.accuracies.append(self.__compute_accuracy())

    def __compute_accuracy(self):
        correct = 0
        for batch_data, batch_targets in self.loader:
            batch_data = batch_data.to(self.device)
            batch_targets = batch_targets.to(self.device)

            predictions = self.model.forward(batch_data)
            classifications = predictions.argmax(dim=-1,
                    keepdim=True).view_as(batch_targets)
            correct += classifications.eq(batch_targets).sum().item()
        accuracy = correct / len(self.dataset) * 100
        return accuracy

