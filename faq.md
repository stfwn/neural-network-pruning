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
