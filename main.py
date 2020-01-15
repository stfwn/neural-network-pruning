#! /bin/python
import argparse
import os
import sys
from datetime import datetime as dt

import torch

from training.Trainer import Trainer
from testing.Tester import Tester
from models.LeNet import LeNet
from pruning import *
# from models.Conv2 import Conv2
# from models.Conv6 import Conv6

def main(args):
    if torch.cuda.is_available() and not args.disable_cuda:
        print('Using cuda.')
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('Using cpu.')
        device = torch.device('cpu')

    # Init model
    if args.model.lower() == 'lenet':
        model_name = 'lenet'
        model = LeNet(device=device)
    elif args.model.lower() == 'conv6':
        model_name = 'conv6'
        model = Conv6(device=device)
    else:
        raise ValueError(f'Model "{args.model}" not supported.')

    if args.dataset.lower() == 'mnist':
        dataset = 'MNIST'
    else:
        raise ValueError(f'Dataset "{args.dataset}" not supported.')

    if args.load_last_pretrained:
        filename = [x for x in sorted(os.listdir('./models/states/')) if
                model_name in x][-1]
        print(f'Loading last pretrained model from ./models/states/{filename}')
        model.load_state_dict(torch.load(f'./models/states/{filename}'))
        model.eval()
        tester = Tester(model, dataset, device=device)
        tester.test_epoch()
        sys.exit(0)

    # Train/test loop
    trainer = Trainer(model, dataset, batch_size=args.batch_size, device=device,
            pruning_rate=args.pruning_rate, pruning_interval=args.pruning_interval)
    tester = Tester(model, dataset, device=device)
    for i in range(args.epochs):
        print(f'Epoch {i}')
        # Train
        model.train()
        trainer.train_epoch(i)

        if args.pruning_rate > 0:
            print(f'Sparsity : {get_sparsity(trainer.model)}')

        # Test
        model.eval()
        tester.test_epoch()

    train_losses, train_accuracies = trainer.losses, trainer.accuracies
    test_losses, test_accuracies = tester.losses, tester.accuracies

    # Save model
    if not args.forget_model:
        now = dt.now().strftime('%Y-%m-%d-%H-%M')
        os.makedirs('./models/states/', exist_ok=True)
        torch.save(model.state_dict(), f'./models/states/{model_name}-{now}.pt')

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
    parser.add_argument('--forget-model', required=False, action='store_true')
    parser.add_argument('--disable-cuda', required=False, action='store_true')
    parser.add_argument('--pruning-rate', type=float, required=False, default=0)
    parser.add_argument('--pruning-interval', type=int, required=False, default=0)
    args = parser.parse_args()
    if args.pruning_rate > 1:
        raise ValueError('Pruning rate cannot be > 1.')
    if args.pruning_rate > 0 and args.pruning_interval == 0:
        raise ValueError('Pruning interval of 0 makes no sense.')
    if args.pruning_rate == 0 and args.pruning_interval > 0:
        raise ValueError('Pruning rate of 0 makes no sense.')

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
