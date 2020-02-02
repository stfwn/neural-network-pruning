# Kunstmatige Intelligentie: Leren en Beslissen

## Usage
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

Run `python main.py --help` to display these instructions locally. Here is a
list of options for parameters that are not self-explanatory. For each of the
initialization schemes there are logs of 11 differently seeded runs with the
LeNet model already present in the `logs` folder.

| Parameter          | Options                                                                                                                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--model`          | `conv6`, `lenet`                                                                                                                                                                              |
| `--dataset`        | `mnist`, `cifar10`                                                                                                                                                                            |
| `--initialization` | `normal`, `uniform`, `kaiming-normal`, `kaiming-uniform`, `xavier-normal`, `xavier-uniform`, `xavier-normal-half`, `xavier-normal-double`, `xavier-uniform-half` and `xavier-uniform-double`. |
| `--seed`           | `int` to seed the Pytorch random number generator with, in order to produce repeatable results.                                                                                                             |


## Directory Structure

```
.
├── README.md
├── main.py                         Entrypoint
├── data                            Data dir (managed)
├── experiments.md                  Notes about experiments
├── faq.md                          Notes about why we did things
├── google-colab-playground.ipynb   Python notebook to use this repo with Google Colab
├── log.md                          Log for university
├── todo.md                         Living to-do list for the project
├── logs
│   ├── <*.log>                     Contain hyperparams and data about runs
│   ├── analyze.py                  Playground to plot things in
│   ├── helpers.py                  Helpers to import/filter run logs
│   └── summarize.py                Script to inspect what logs are present in this folder
├── tensorboard-logs                Folder where TensorBoard logs are written to
├── report                          Folder containing everything for the final report
├── slides                          Folder containing everything for presentations
├── models
│   ├── ExpandedModule.py           Base class with weight saving/resetting functionality
│   ├── Conv6.py                    Convolutional network model with 6 conv layers
│   └── LeNet.py                    Fully connected neural network model based on LeNet
├── initializing.py                 Code for different weight init schemes
├── pruning.py                      Pruning code (by Andrei @ BrainCreators)
├── testing                         Code for testing networks
│   └── Tester.py
└── training                        Code for training networks
    └── Trainer.py
```
