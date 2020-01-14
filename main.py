#! /bin/python
import argparse

from training.train import train
from models.LeNet import LeNet

def main(args):
    model = LeNet()
    train(model, 'MNIST')

def parse_args():
    parser = argparse.ArgumentParser(description='Entrypoint for training/testing models in this repository.')

    parser.add_argument('-p', '--load-pretrained', type=str,
            help='Load a pretrained model from ./models/states')
    parser.add_argument('--load-pretrained-last', type=bool, default=False,
            help='Load most recently saved model in ./models/states.')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
