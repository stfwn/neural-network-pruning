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
├── models
│   ├── __init__.py
│   ├── LeNet.py
│   └── states                              Where trained models are stored.
│       └── 2020-01-14-18-03.pt
├── README.md
├── results                                 Where results will be stored.
├── testing                                 Contains testing code.
│   ├── __init__.py
│   └── Tester.py
├── todo.md
└── training                                Contains training code.
    ├── __init__.py
    └── Trainer.py
```
