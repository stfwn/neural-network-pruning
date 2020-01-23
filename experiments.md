# Experiments

2 models x 4 init schemes x 3 seeds = 24 runs

| Param              | Different values           |
|--------------------|----------------------------|
| Seed               | 42, 43, 44                 |
| Model/Dataset pair | LeNet/MNIST, Conv6/CIFAR10 |
| Runs               | 5 x each                   |
| Init schemes       | Xavier, Kaiming            |

- [x] Question: Xavier & Kaiming normal/uniform dis different runs? __Answer:__ yes.
- [x] Question: test with varying pruning settings or not needed? __Answer:__
  use one setting for all runs. If we have time, do more runs with different
  settings.
- [x] Check pruning rates and intervals for both model/dataset combinations in
  paper. __Answer:__ Use 5 epochs as interval (enough to recover), 20% as rate.

```bash
# Invoke the program
python main.py

# Set seed
-s 42
-s 43
-s 44
-s 45

# LeNet/MNIST runs should always have these options:
-m lenet -d mnist -l 1.2e-3 -b 60 -e 100

# Conv6/CIFAR10 runs should always have these options:
-m conv6 -d cifar10 -l 3e-4 -b 60 -e 100

# The init scheme option is ONE of:
-i xavier-uniform
-i xavier-normal
-i kaiming-uniform
-i kaiming-normal

# Additionally, the pruning params
--pruning-rate .2 --pruning-interval 5
```

| Who    | Seed |
|--------|------|
| Ellis  | 42   |
| Stefan | 43   |
| Thomas | 44   |
| Stefan | 45   |
