#! /bin/python
import argparse
import os

import torch

from training.train import train
from testing.test import test
from models.LeNet import LeNet
# from models.Conv2 import Conv2

def main(args):
    if args.model.lower() == 'lenet':
        model = LeNet()
    elif args.model.lower() == 'conv2':
        print('Placeholder to init Conv2 model.')
        # model = Conv2()
        raise ValueError
    if args.load_last_pretrained:
        # TODO: make distinction between different models when we have
        # different models. Otherwise this will bug out.
        filename = sorted(os.listdir('./models/states/'))[0]
        print(f'Loading last pretrained model from ./models/states/{filename}')
        model.load_state_dict(torch.load(f'./models/states/{filename}'))
    else:
        print('Training model.')
        train(model, 'MNIST')

    print('Testing model.')
    model.eval()
    test(model, 'MNIST')

def parse_args():
    parser = argparse.ArgumentParser(description='Entrypoint for training/testing models in this repository.')

    # TODO: Add functionality to load specific pretrained model.
    #parser.add_argument('-p', '--load-pretrained', type=str,
    #        help='Load a pretrained model from ./models/states/')
    parser.add_argument('--load-last-pretrained', action='store_true',
            help='Load most recently saved model in ./models/states/.')
    parser.add_argument('-m' , '--model', type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
