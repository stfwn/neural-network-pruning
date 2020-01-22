# FAQ

## Why not write your own Dataset or VisionDataset class for the datasets?

MNIST comes in a binary format (idx) that requires a custom parsing function.
The format is relatively simple; the function to do it [looks like
this](https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py#L431-L457q).
Although it is useful to learn about the Pytorch dataset classes, it is not
relevant to the topic of our internship to reimplement this method.

## Why are you using the hyperparameters that you are using?

For the FC model for MNIST, we use the LeNet 300-100 architecture from LeCun
(1998) with the optimizer and hyperparameters that Frankle & Carbin (2019) used
in their paper about Lottery Tickets. This allows us to compare results and
draw a conclusion about whether or not looking into different initialization
strategies is worth it.

## In Morcos et al. late resetting is used, why do you not do this?

Late resetting means that when resetting weights after pruning you do not reset
them to their _initial_ value but to their value after _k_ epochs of training.
We decided not to use late resetting in our research so that the effects of
different initialization strategies -- if any -- would be more pronounced, not
diluted by one optimization step.

Note: perhaps mention paper "Stabilizing the lottery ticket hypothesis" where
this came from (source: Andrei) in relation with big data sets, which we do not
have here.
