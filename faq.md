# FAQ
## Why not write your own Dataset or VisionDataset class for the datasets?
MNIST comes in a binary format (idx) that requires a custom parsing function.
The format is relatively simple; the function to do it [looks like
this](https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py#L431-L457q).
Although it is useful to learn about the Pytorch dataset classes, it is not
relevant to the topic of our internship to reimplement this method.
