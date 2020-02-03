---
title: One Initialization to Rule Them All?
author: Stefan Wijnja, Ellis Wierstra, Thomas Hamburger
date: \today{}
abstract: |
    As universal function approximators, neural networks have proven to be a
    useful tool in many areas of research and industry. However, the
    computational efficiency of deep learning leaves much to be desired. On
    this topic, @frankle2019 have presented the lottery ticket hypothesis,
    which states that on initialization, large networks contain weight
    combinations that make up subnetworks that can rival the larger versions in
    performance when trained in isolation. In this paper a comparison between
    different weight initialization algorithms is made to find out how a
    network's starting point affects the search for these subnetworks. Results
    point toward a pattern where drawing weights from Xavier-based, narrower
    distributions may outperform drawing from Kaiming-based, wider
    distributions on this task, while the difference in shape of a normal
    versus uniform distribution may not matter much. More research is needed to
    confirm or falsify these preliminary conclusions.
toc: true
margin-left: 4.8cm
margin-right: 4.8cm
---

## Introduction

As universal function approximators, neural networks have proven to be a useful
tool in many areas of research and industry. To meet the challenge of modeling
complex functions, network sizes -- in terms of the number of parameters --
have increased as well, and these large networks require significant amounts of
computational power and time to train and use.

On this topic, @frankle2019 have presented the Lottery Ticket Hypothesis, which
states:

> A randomly initialized, dense neural-network contains a subnetwork that is
> initialized such that -- when trained in isolation -- it can match the test
> accuracy of the original network after training for at most the same number
> of iterations. [@frankle2019]

It has been shown before that by methodically removing weights from a network
(_pruning_), network sizes can be reduced by up to 90% without loss in
performance [@frankle2019: 1]. The novelty here lies in the idea that a reduced
network is not simply used as-is, but its weights are reset to the values
post-initialization and then retrained. 

Indeed, @frankle2019's research has shown that the reduced and retrained
networks can be up to 96 percent smaller than the original, full-sized network
without loss in test accuracy [@frankle2019: 4]. This proves that training the
full network is only relevant in _finding_ the configuration of weights that
make up the performant subnetwork -- the winning ticket --, not in training and
using it.

Naturally this leads one to wonder if there is a way to skip training the large
network and initialize the winning ticket in one shot. To inch closer to this
ideal, we compare different initialization algorithms to see how a network's
starting point affects the search for winning tickets.
