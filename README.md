# Kunstmatige Intelligentie: Leren en Beslissen

## Usage
Run `python main.py --help` for help.See the bottom of `initializing.py` for a dictionary of which init schemes are
supported at this time.

```
usage: main.py [-h] [--load-last-pretrained] -m MODEL -d DATASET [-e EPOCHS]
               [-b BATCH_SIZE] [-i INITIALIZATION] [-l LEARNING_RATE]
               [--forget-model] [--disable-cuda] [--pruning-rate PRUNING_RATE]
               [--pruning-interval PRUNING_INTERVAL] [-s SEED]

Entrypoint for training/testing models in this repository.

optional arguments:
  -h, --help            show this help message and exit
  --load-last-pretrained
                        Load most recently saved model in ./models/states/.
  -m MODEL, --model MODEL
  -d DATASET, --dataset DATASET
  -e EPOCHS, --epochs EPOCHS
  -b BATCH_SIZE, --batch-size BATCH_SIZE
  -i INITIALIZATION, --initialization INITIALIZATION
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
  --forget-model
  --disable-cuda
  --pruning-rate PRUNING_RATE
  --pruning-interval PRUNING_INTERVAL
  -s SEED, --seed SEED
```

Here is a list of options for parameters that are not self-explanatory:

| Parameter          | Options                                                                                                                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--model`          | `conv6`, `lenet`                                                                                                                                                                              |
| `--dataset`        | `mnist`, `cifar10`                                                                                                                                                                            |
| `--initialization` | `normal`, `uniform`, `kaiming-normal`, `kaiming-uniform`, `xavier-normal`, `xavier-uniform`, `xavier-normal-half`, `xavier-normal-double`, `xavier-uniform-half` and `xavier-uniform-double`. |
| `--seed`           | `int` to seed the Pytorch random number generator with, in order to produce repeatable results.                                                                                                             |


## Directory Structure

```
.
├── data                            Data dir, managed by program.
├── experiments.md                  Notes about experiments.
├── faq.md                          Notes about why we did things.
├── google-colab-playground.ipynb   Python notebook to use this repo with Google Colab.
├── initializing.py                 Code for different weights init schemes.
├── log.md                          Log for university.
├── logs
│   ├── <*.log>                     Contain hyperparams and data about runs.
│   ├── analyze.py                  Playground to plot things in.
│   ├── helpers.py                  Helpers to import/filter run logs.
│   └── summarize.py                Script to inspect what logs are present in this folder.
├── main.py
├── models
│   ├── Conv6.py                    Convolutional network model with 6 conv layers.
│   ├── ExpandedModule.py           Base class with weight saving/resetting functionality.
│   └── LeNet.py                    Fully connected neural network model based on LeNet.
├── pruning.py                      Pruning code (by Andrei @ BrainCreators).
├── README.md
├── report                          Folder containing everything for the final report.
├── slides                          Folder containing everything for presentations.
├── testing                         Code for testing networks.
│   └── Tester.py
├── todo.md                         To-do list for our project.
└── training                        Code for training networks.
    └── Trainer.py
```
