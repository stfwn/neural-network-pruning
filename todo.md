# Todo

## Week 1: Getting Started
### 2020-01-07
- [x] Write initial todo.

### 2020-01-10
- [x] Send information to Nicolaas Heyning.
    - [ ] Ellis
    - [x] Stefan
    - [x] Thomas
- [x] Read about neural networks: https://www.3blue1brown.com/neural-networks
    - [X] Ellis
    - [x] Stefan
    - [x] Thomas
- [x] Understand Pytorch example: https://colab.research.google.com/drive/1arq7ZpWoO4Xw1od_RbMTHCl5IwIoYXOZ
    - [X] Ellis
    - [X] Stefan
    - [x] Thomas
- [x] Papers en blogs over pruning lezen.
    - [X] Ellis
    - [X] Stefan
    - [x] Thomas

## Week 2: Getting Pytorch Down
### 2020-01-14
- [x] Step 1 & 2.
    - [x] 1: Implement a feedforward neural network and train it on the MNIST dataset.
    - [x] 2: Implement a convolutional neural network and train it on the MNIST dataset.
    - [x] When done, send Andrei mail for pruning code.
    - [ ] Optional: do the same for the CIFAR10 dataset.

### 2020-01-16
- [x] Step 3.
    - [x] Incorporate pruning mechanism.
    - [x] Implement resetting function.

## Week 3: Perform Experiments
### 2020-01-21
- [ ] Step 4.
    - [ ] Experiment with different initialization schemas.
          - [ ] Pick different schema's to try, figure out what they are
                and decide who does which one.
                - [ ] Every schema is to be experimented with a total of 8 times. 
                      Do this with certain rates for pruning rate and pruning interval.
                      Make sure to document which models are result of which initialisation and what variables.
          - [ ] Answer the question: does the initialization schema of a
                network affect its robustness to pruning? (Compare the results of the different initialions and
                draw a conclusion)

### 2020-01-23
- [ ] Step 5.
    - [ ] Determine for both of the above networks, and at varying degrees of
          sparsity levels, which schema works best and why.

## Week 4: Do Optional Things and Write Report
### 2020-01-28
- [ ] Step 6 & 7.
    - [ ] Based on the above results, is it possible, then, to construct a
          custom initialization schema that improves the robustness further?
          - [ ] Compare the different initialisations. Is there a major difference, and if so, what component
                is responsible? Is it usable for a custom initialisation schema?
          - [ ] If it is possible to construct a custom schema, let's experiment with it. Does it improve robustness?
    - [ ] Check whether the schema built on (6) generalizes to other pruning methods.

### 2020-01-28
- [ ] Presentations
    - [ ] Powerpoint (?)
    - [ ] Text

### 2020-01-31
- [ ] Final Report
