#! /bin/python
import argparse
import os
import sys
from datetime import datetime as dt

import torch

from training.Trainer import Trainer
from testing.Tester import Tester
from models.LeNet import LeNet
# from models.Conv2 import Conv2

def main(args):
    # Init model
    if args.model.lower() == 'lenet':
        model = LeNet()
    elif args.model.lower() == 'conv2':
        # model = Conv2()
        raise ValueError('This is still a placeholder.')

    if args.dataset.lower() == 'mnist':
        dataset = 'MNIST'
    else:
        raise ValueError(f'Dataset "{args.dataset}" not supported.')

    if args.load_last_pretrained:
        # TODO: make distinction between different models when we have
        # different models. Otherwise this will bug out.
        filename = sorted(os.listdir('./models/states/'))[-1]
        print(f'Loading last pretrained model from ./models/states/{filename}')
        model.load_state_dict(torch.load(f'./models/states/{filename}'))
        model.eval()
        test(model, dataset)
        sys.exit(0)

    # Train/test loop
    trainer = Trainer(model, dataset, args.batch_size)
    tester = Tester(model, dataset)
    for i in range(args.epochs):
        print(f'Epoch {i}')
        # Train
        model.train()
        trainer.train_epoch()

        # Test
        model.eval()
        tester.test_epoch()

    train_losses, train_accuracies = trainer.losses, trainer.accuracies
    test_losses, test_accuracies = tester.losses, tester.accuracies

    # Save model
    if args.save_model:
        now = dt.now().strftime('%Y-%m-%d-%H-%M')
        os.makedirs('./models/states/', exist_ok=True)
        torch.save(model.state_dict(), f'./models/states/{now}.pt')

    # TODO: replace with plot and save shizzle.
    print(f'train_losses={train_losses}')
    print(f'train_accuracies={train_accuracies}')
    print(f'test_losses={test_losses}')
    print(f'test_accuracies={test_accuracies}')

def parse_args():
    parser = argparse.ArgumentParser(description='Entrypoint for training/testing models in this repository.')

    # TODO: Add functionality to load specific pretrained model.
    #parser.add_argument('-p', '--load-pretrained', type=str,
    #        help='Load a pretrained model from ./models/states/')
    parser.add_argument('--load-last-pretrained', action='store_true',
            help='Load most recently saved model in ./models/states/.')
    parser.add_argument('-m' , '--model', type=str, required=True)
    parser.add_argument('-d' , '--dataset', type=str, required=True)
    parser.add_argument('-e', '--epochs', type=int, required=False, default=50)
    parser.add_argument('-b', '--batch-size', type=int, required=False, default=60)
    parser.add_argument('-s', '--save-model', type=bool, required=False, default=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
