---
title: Making Networks Prune-Ready
author: Stefan Wijnja, Ellis Wierstra, Thomas Hamburger
date: \today{}
abstract: TODO
toc: true
---

## Introduction

* TODO: Neural networks are cool

Neural networks are possibly the technological development with the most potential 
in the future. Neural networks can be trained to carry out tasks like object detection 
or shape recognition. These functions can be very useful in every day life, for shape 
recognition or self driving cars.

* TODO: But they can be very demanding

But sadly there is a flip side to these functions of neural networks. Getting high 
performance on these tasks comes with a pretty high price. Getting a high performance 
on these tasks typically comes with a high number of parameters within the network, 
which demand great computational and memory resources.

A way to counter this demand for resources is to create sparser networks, meaning 
smaller by lessening the amount of connections within the network. A way to do this
is by using the Lottery Ticket Hypothesis by Frankle and Carbin, 2019.

* TODO: Enter: the Lottery Ticket Hypothesis

> The Lottery Ticket Hypothesis: A randomly initialized, dense neural-network
> contains a subnetwork that is initalized such that -- when trained in
> isolation -- it can match the test accuracy of the original network after
> training for at most the same number of iterations. [@frankle2019]

__Iterative, magnitude-based pruning.__ In general, pruning is the idea of
eliminating weights from a neural network, producing a smaller network.
Magnitude-based pruning specifies that the weights that will be nullified are
those in the top $X$ percent of weights closest to $0$ -- the intuition being
that the weights that are smallest in absolute terms are the least relevant to
the network's outcome and should be the first to go. Iterative, magnitude-based
pruning then describes a process where not all pruning is done in one step, but
over the course of multiple, spread-out pruning steps. [@frankle2019]

* TODO: Status quo on pruning

* TODO: Our question: how do initialization algorithms compare when looking for
  winning tickets?
