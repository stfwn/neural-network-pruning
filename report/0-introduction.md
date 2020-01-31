---
title: Making Networks Prune-Ready
author: Stefan Wijnja, Ellis Wierstra, Thomas Hamburger
date: \today{}
abstract: TODO
toc: true
---

## Introduction

* TODO: Neural networks are cool

* TODO: But they can be very demanding

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
