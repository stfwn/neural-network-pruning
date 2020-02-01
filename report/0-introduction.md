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
possibility of neural networks to learn any function regardless of
complexity [@scarselli1998] contributes to their growing importance. This
universality makes deep neural networks useful in a wide range of fields, for
instance medical imaging, stock price prediction and natural disaster
forecasting. The complexity of the underlying function to be learned is strongly
related to the minimally required size of the neural network.
[@lecun1989generalization].

An important disadvantage of large neural networks is that they contain
a large number of parameters and thus require great amounts of computing power
and memory resources to be trained. Moreover, complex networks have a
tendency to overfit to the problem they are trained on, which lowers their
generalizability to other problems [@lecun1989generalization], while networks
that are too small, do not have the power and flexibility to represent the data.
[@lecun1990].

__Pruning__
To solve some of the problems associated with large neural networks, the density
of a neural network can be reduced by cutting down the number of nonzero-valued
parameters in its weight matrices, thereby inducing sparsity. This process is
called pruning [@lecun1990]. It has been shown that pruning can reduce the size
of neural networks in terms of parameter count by more than 90% [@lecun1990].

One issue the with pruning a large network that training the complete network is
still computationally intensive [@lecun1989generalization].
But if we can reduce the trained network in size, would it be possible to
start with the pruned network and train it from the start, thereby reducing
the computational complexity of the training process?

__Enter: the Lottery Ticket Hypothesis__ Based on the the observation that the
configuration of the starting weights of a pruned network affects its capacity
to learn, [@frankle2019] proposed the Lottery Ticket Hypothesis:

> The Lottery Ticket Hypothesis: A randomly initialized, dense neural-network
> contains a subnetwork that is initalized such that -- when trained in
> isolation -- it can match the test accuracy of the original network after
> training for at most the same number of iterations. [@frankle2019]

The specific subnetwork that learns particularly well is called a "winning
ticket". These networks converge faster and have better performance in terms of
test accuracy.

The initial values of the weights are drawn from a random sample. But the
specific distributions these samples are drawn from differ between
initialization methods. This raises the question if differently initialized
networks differ in their performance after pruning. In this project we would
like to uncover how initialization algorithms compare when looking for winning
tickets.

Although training highly sparse networks is far more efficient computationally,
finding these winning ticket initializations still requires training the
complete network which is computationally expensive. However, [@morcos2019]
have found that winning tickets can be reused for other datasets
and optimizers. This increases their utility, because the computationally
expensive task of finding the winning ticket only has to be done once.

__Iterative, magnitude-based pruning.__
Magnitude-based pruning specifies that the weights that will be nullified are
those in the top $X$ percent of weights closest to $0$ -- the intuition being
that the weights that are smallest in absolute terms are the least relevant to
the network's outcome and should be the first to go. Iterative, magnitude-based
pruning then describes a process where not all pruning is done in one step, but
over the course of multiple, spread-out pruning steps. [@frankle2019]
