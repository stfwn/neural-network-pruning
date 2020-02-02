## Appendix A: Extra Information of Practical Interest

__Code and results.__ We welcome efforts to reproduce and extend our work, and
would be happy to get in touch about sharing our repository of code and
results. Please contact Stefan Wijnja at s.wijnja@me.com for any queries.

__PyTorch.__ The neural networks in this project are built using the open
source machine learning package PyTorch[^1]. Among other things, PyTorch
includes modules for defining network models, doing backpropagation using
automatic differentiation, various optimizers and some built-in datasets. These
modules greatly reduce the amount of code and time required to build and train
the networks, leaving more time for experiments and analysis of the results.

[^1]: https://pytorch.org/

__Randomization.__ As noted in the report itself, 11 experiments were done for
each combination of factors in order to reduce noise in the results. In the
interest of reproducibility it is useful for you to know that the seeds used
range from 42 to 52 (inclusive).
