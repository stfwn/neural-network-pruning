# Experiments

2 models x 2 init schemes x 3 seeds x 5 runs = 60 runs.

| Param              | Different values           |
|--------------------|----------------------------|
| Seed               | 42, 43, 44                 |
| Model/Dataset pair | LeNet/MNIST, Conv6/CIFAR10 |
| Runs               | 5 x each                   |
| Init schemes       | Xavier, Kaiming            |

TODO: Question: Xavier & Kaiming normal/uniform dis different runs?
TODO: QUESTION: Varying pruning rate or nah?
TODO: Find learning rate & pruning rate for both model/shizzle combinations in paper

```bash
# Invoke the program
python main.py

# Set seed
-s 69
-s 42
-s 420

# LeNet/MNIST runs should always have these options:
-m lenet -d mnist -l 1.2e-3 -b 60 -e 50

# Conv6/CIFAR10 runs should always have these options:
-m conv6 -d cifar10 -l 3e-4 -b 60 -e 30

# The init scheme option is ONE of:
-i xavier-uniform
-i xavier-normal
-i kaiming-uniform
-i kaiming-normal

# Additionally, the pruning params
--pruning-rate ??? --pruning-interval ???
```

| Who    | Seed |
|--------|------|
| Ellis  | 42   |
| Stefan | 43   |
| Thomas | 44   |
