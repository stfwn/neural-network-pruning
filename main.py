#! /bin/python
import argparse
import os
import sys
from datetime import datetime as dt

import torch
import torch.nn as nn

from training.Trainer import Trainer
from testing.Tester import Tester
from models.LeNet import LeNet
from models.Conv6 import Conv6

from pruning import *
from initializing import init_weights

from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np

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

    # Create log file and add params to directory and as text to tensorboard
    now = dt.now().strftime('%Y-%m-%d-%H-%M')
    writer = SummaryWriter(log_dir=f"./tensorboard-logs/{model_name}_{dataset.lower()}_i{args.initialization}_pr{args.pruning_rate}_pi{args.pruning_interval}_e{args.epochs}_b{args.batch_size}_s{args.seed}/{now}")
    writer.add_text('hparams', json.dumps(vars(args)))

    # Train/test loop
    trainer = Trainer(model, dataset, batch_size=args.batch_size, device=device,
            pruning_rate=args.pruning_rate,
            pruning_interval=args.pruning_interval,
            learning_rate=args.learning_rate)
    tester = Tester(model, dataset, device=device)
    sparsities = []
    for i in range(args.epochs):
        print(f'======= Epoch {i} ======= =======')
        # Train
        model.train()
        trainer.train_epoch(i)

        # Test
        model.eval()
        tester.test_epoch()
        sparsities.append(get_sparsity(model))
        print(f'\taccuracy\tloss\n' +
                f'train\t{trainer.accuracies[-1]:.5f}\t{trainer.losses[-1]:.5f}\n' +
                f'test\t{tester.accuracies[-1]:.5f}\t{tester.losses[-1]:.5f}\n' +
                f'sparsity:{sparsities[-1]:.5f}')

        # Write Tensorboard log
        log(writer, model, tester, trainer, i)

    # Save model
    if not args.forget_model:
        now = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs('./models/states/', exist_ok=True)
        torch.save(model.state_dict(), f'./models/states/{model_name}-{now}.pt')

    # Save simple version of logs
    simple_log(args, trainer, tester, sparsities)


def log(writer, model, tester, trainer, i):
    writer.add_scalar('acc/train', trainer.accuracies[-1], i)
    writer.add_scalar('acc/test', tester.accuracies[-1], i)
    writer.add_scalar('loss/train', trainer.losses[-1], i)
    writer.add_scalar('loss/test', tester.losses[-1], i)
    writer.add_scalar('sparsity/sparsity', get_sparsity(model), i)
    if args.model.lower() == 'lenet':
        all_weights = torch.cat([layer.weight.flatten() for layer in model.layers if type(layer) is nn.Linear or type(layer) is nn.Conv2d], axis=0)
        all_biases = torch.cat([layer.bias.flatten() for layer in model.layers if type(layer) is nn.Linear or type(layer) is nn.Conv2d], axis=0)
        all_gradients = torch.cat([layer.weight.grad.flatten() for layer in model.layers if type(layer) is nn.Linear or type(layer) is nn.Conv2d], axis=0)
    else:
        conv_weights = torch.cat([layer.weight.flatten() for layer in model.conv if type(layer) is nn.Conv2d], axis=0)
        lin_weights = torch.cat([layer.weight.flatten() for layer in model.fc if type(layer) is nn.Linear], axis=0)
        all_weights = torch.cat([conv_weights, lin_weights], axis=0)

        conv_biases = torch.cat([layer.bias.flatten() for layer in model.conv if type(layer) is nn.Conv2d], axis=0)
        lin_biases = torch.cat([layer.bias.flatten() for layer in model.fc if type(layer) is nn.Linear], axis=0)
        all_biases = torch.cat([conv_biases, lin_biases], axis=0)

        conv_gradients = torch.cat([layer.weight.grad.flatten() for layer in model.conv if type(layer) is nn.Conv2d], axis=0)
        lin_gradients = torch.cat([layer.weight.grad.flatten() for layer in model.fc if type(layer) is nn.Linear], axis=0)
        all_gradients = torch.cat([conv_gradients, lin_gradients], axis=0)

    writer.add_histogram('weight', all_weights, i)
    writer.add_histogram('bias', all_biases, i)
    writer.add_histogram('weight.gradient', all_gradients, i)

def simple_log(args, tester, trainer, sparsities):
    log = {'args': vars(args),
            'test_acc': tester.accuracies,
            'test_loss': tester.losses,
            'train_acc': trainer.accuracies,
            'train_loss': trainer.losses,
            'sparsity': sparsities}

    os.makedirs('./logs/', exist_ok=True)
    now = dt.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    logfiles = os.listdir('./logs/')
    filename = now + '.log'
    while filename in logfiles:
        now = dt.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
        logfiles = os.listdir('./logs/')
        filename = now + '.log'
    with open(f'./logs/{filename}', 'w') as fp:
        fp.write(json.dumps(log, indent=4))


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
    parser.add_argument('-s', '--seed', type=int, default=42)
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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
