# Kunstmatige Intelligentie: Leren en Beslissen

## Directory Structure

```
.
├── data                                    Created by PyTorch, houses data.
│   └── MNIST
│       ├── processed
│       │   ├── test.pt
│       │   └── training.pt
│       └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-labels-idx1-ubyte
│           ├── train-images-idx3-ubyte
│           └── train-labels-idx1-ubyte
├── faq.md                                  Reasoning about why we did things.
├── log.md
├── main.py                                 CLI entrypoint.
├── models                                  Where models are stored.
│   ├── __init__.py
│   ├── LeNet.py
│   └── states                              Where net states are stored.
│       ├── 2020-01-14-11-58.pt
│       └── 2020-01-14-12-09.pt
├── README.md                               This file.
├── testing                                 Code to test nets.
│   ├── __init__.py
│   └── test.py
├── todo.md
└── training                                Code to train nets.
    ├── __init__.py
    └── train.py

9 directories, 27 files
```
