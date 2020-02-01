---
title: Making Networks Prune-Ready
author: Stefan Wijnja, Ellis Wierstra, Thomas Hamburger
date: \today{}
abstract: TODO
toc: true
---

## Introduction

The developments in the field of deep learning have recently been accelerating,
in part due to advancements in hardware capabilities. The theoretical
possibility of neural networks to learn any function, regardless of its
complexity [@scarselli1998] also contributes to their widespread use. This
universality makes deep neural networks suitable for use in fields such as
medical imaging, stock price prediction and natural disaster forecasting.
The complexity of the function to be learned is strongly related to the required
minimally required size of the neural network which is capable of learning it
[@lecun1989generalization].

An important disadvantage of highly complex neural networks is that they contain
a large number of parameters and thus require great amounts of computing power
and memory resources to be trained. In addition, complex networks have a
tendency to overfit to the specific problem they are trained, which lowers their
generalisability to other problems [@lecun1989generalization]. While networks
that are too small, do not have the power and flexibility to represent the data
[@lecun1990].

The density of a neural network can be reduced by cutting down the number of
nonzero-valued parameters in its weight matrices, inducing sparsity. This
process is called pruning [@lecun1990]. It has been shown that pruning can
reduce the size of neural networks in terms of parameter count by more than 90%
[@lecun1990].
Training the complete network and then pruning the trained network has the
problem that training a large network is still computationally intensive
[@lecun1989generalization].
However, if we can reduce the trained network in size, would it be possible to
start with the pruned network and train it from the beginning, thereby reducing
the computational complexity of the training process?
Unfortunately, it has been shown that training the pruned networks is harder and
leads to lower performance than training the original network [@han2015]. This
has raised the question if differently initialized networks differ in their
performance after pruning.

__Enter: the Lottery Ticket Hypothesis__ Based on the the observation that the
configuration of the starting weights of a pruned network affects its capacity
to learn, [@frankle2019] proposed the Lottery Ticket Hypothesis:
>>>>>>> Stashed changes

> The Lottery Ticket Hypothesis: A randomly initialized, dense neural-network
> contains a subnetwork that is initalized such that -- when trained in
> isolation -- it can match the test accuracy of the original network after
> training for at most the same number of iterations. [@frankle2019]

The specific subnetwork that learns particularly well is called a "Winning
Ticket". These networks converge faster and have better performance in terms of
test accuracy.

Although training highly sparse networks is far more efficient computationally,
finding these winning ticket initialisations still requires training the
complete network which is computationally expensive.
[@morcos2019] have found that winning tickets can be reused for other datasets
and optimizers. This increases their utility, because the computationally
expensive task of finding only has to be done once.

The initial values of the weights are typically drawn from a random sample. The
specific distributions these samples are drawn from differ from method to
method.
In this project we would like to uncover how initialisation algorithms compare
when looking for winning tickets.

__Iterative, magnitude-based pruning.__ In general, pruning is the idea of
eliminating weights from a neural network, producing a smaller network.
Magnitude-based pruning specifies that the weights that will be nullified are
those in the top $X$ percent of weights closest to $0$ -- the intuition being
that the weights that are smallest in absolute terms are the least relevant to
the network's outcome and should be the first to go. Iterative, magnitude-based
pruning then describes a process where not all pruning is done in one step, but
over the course of multiple, spread-out pruning steps. [@frankle2019]
