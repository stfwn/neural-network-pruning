#! /bin/python
import argparse
import os
import sys
from datetime import datetime as dt

import torch

from training.Trainer import Trainer
from testing.Tester import Tester
from models.LeNet import LeNet
from models.Conv6 import Conv6

from pruning import *
from initializing import init_weights

from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(args):
    # Check for cuda
    if torch.cuda.is_available() and not args.disable_cuda:
        print('Using cuda.')
        device = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print('Using cpu.')
        device = torch.device('cpu')

    # Init model itself
    if args.model.lower() == 'lenet':
        model_name = 'lenet'
        model = LeNet(device=device)
    elif args.model.lower() == 'conv6':
        model_name = 'conv6'
        model = Conv6(device=device)
    else:
        raise ValueError(f'Model "{args.model}" not supported.')

    # Apply alternative init method if instructed to do so 
    if args.initialization:
        init_weights(model, args.initialization)

    # Init dataset
    if args.dataset.lower() == 'mnist':
        dataset = 'MNIST'
    elif args.dataset.lower() == 'cifar10':
        dataset = 'CIFAR10'
    else:
        raise ValueError(f'Dataset "{args.dataset}" not supported.')

    # Load last pretrained, test and exit if instructed to do so.
    if args.load_last_pretrained:
        filename = [x for x in sorted(os.listdir('./models/states/')) if
                model_name in x][-1]
        print(f'Loading last pretrained model from ./models/states/{filename}')
        model.load_state_dict(torch.load(f'./models/states/{filename}'))
        model.eval()
        tester = Tester(model, dataset, device=device)
        tester.test_epoch()
        sys.exit(0)
    
    # TODO : add run name
    now = dt.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"./results/{model_name}_{dataset}_epochs={args.epochs}_batch_size={args.batch_size}_initialization={args.initialization}_seed={args.seed}_/{now}")
    writer.add_text('hparams', json.dumps(vars(args)))

    # Train/test loop
    trainer = Trainer(model, dataset, batch_size=args.batch_size, device=device,
            pruning_rate=args.pruning_rate,
            pruning_interval=args.pruning_interval,
            learning_rate=args.learning_rate)
    tester = Tester(model, dataset, device=device)
    for i in range(args.epochs):
        print(f'======= Epoch {i} ======= =======')
        # Train
        model.train()
        trainer.train_epoch(i)

        # Test
        model.eval()
        tester.test_epoch()
        print(f'\taccuracy\tloss\n' +
                f'train\t{trainer.accuracies[-1]:.5f}\t{trainer.losses[-1]:.5f}\n' +
                f'test\t{tester.accuracies[-1]:.5f}\t{tester.losses[-1]:.5f}\n' +
                f'sparsity:{get_sparsity(trainer.model):.5f}')

        # Write Tensorboard log
        log(writer, model, tester, trainer, i)

    # Save model
    if not args.forget_model:
        now = dt.now().strftime('%Y-%m-%d-%H-%M')
        os.makedirs('./models/states/', exist_ok=True)
        torch.save(model.state_dict(), f'./models/states/{model_name}-{now}.pt')


def log(writer, model, tester, trainer, i):
    writer.add_scalar('acc/train', trainer.accuracies[-1], i)
    writer.add_scalar('acc/test', tester.accuracies[-1], i)
    writer.add_scalar('loss/train', trainer.losses[-1], i)
    writer.add_scalar('loss/test', tester.losses[-1], i)
    writer.add_scalar('sparsity/sparsity', get_sparsity(model), i)


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
    parser.add_argument('-i', '--initialization', type=str, required=False)
    parser.add_argument('-l', '--learning-rate', type=float, required=False,
            default=1.2e-3)
    parser.add_argument('--forget-model', required=False, action='store_true')
    parser.add_argument('--disable-cuda', required=False, action='store_true')
    parser.add_argument('--pruning-rate', type=float, required=False, default=0)
    parser.add_argument('--pruning-interval', type=int, required=False, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.pruning_rate > 1:
        raise ValueError('Pruning rate cannot be > 1.')
    if args.pruning_rate > 0 and args.pruning_interval == 0:
        raise ValueError('Pruning interval of 0 makes no sense.')
    if args.pruning_rate == 0 and args.pruning_interval > 0:
        raise ValueError('Pruning rate of 0 makes no sense.')
    if args.model.lower() == 'lenet' and args.dataset.lower() == 'cifar10':
        raise ValueError('Model LeNet is not configured for dataset CIFAR10.')
    if args.model.lower() == 'conv6' and args.dataset.lower() == 'mnist':
        raise ValueError('Model Conv6 is not configured for dataset MNIST.')

    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
