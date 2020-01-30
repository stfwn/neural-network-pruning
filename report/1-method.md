## Method

To find out how initialization algorithms compare when looking for winning
lottery tickets, a large number of experiments were done with the following
specifications.

__Dataset.__ The MNIST dataset as used, described and published in @lecun1998.
This is a labeled collection with $28 \times 28$ grey-scale images of
handwritten digits, $60$k of which were used as a training set, and $10$k as a
testing set.

__Model.__ A neural network following the specifications in @lecun1998, as is
also used in @frankle2019 in their paper about the lottery ticket hypothesis.
It is a fully connected network containing two hidden layers with 300 and 100
neurons, in that order. With an 784-neuron input layer -- sized to the MNIST's
$28 \times 28$ images, this sums up to $266.2$k weights.

__Initialization methods.__ Four initialization algorithms were tested, stemming
from two distinct approaches to the initialization process: one named _Xavier_,
coined by @glorot2010, and one called _Kaiming_, originating from @kaiming2015.
Each approach offers a formula to supply the specifications for both a normal
and a uniform distribution, which is how we arrive at four algorithms in total.

|                                | Xavier                                              | Kaiming                           |
|--------------------------------|-----------------------------------------------------|-----------------------------------|
| $\mathcal{U}(-a, a)$           | $\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$ | $\sqrt{\frac{3}{\text{fan\_in}}}$ |
| $\mathcal{N}(0, \text{std}^2)$ | $\sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$ | $\sqrt{\frac{1}{\text{fan\_in}}}$ |

In the table[^1], the equations specify the bounds of a uniform ($\mathcal{U}$)
or standard deviation of a normal ($\mathcal{N}$) distribution. The intuition
here is that both formulas scale the size of the weights in a layer inversely
proportional to the number connections to that layer, in order to keep the
activation from exploding or vanishing on the forward pass. The Xavier method
incorporates more information about the number of connections to additionally
prevent the gradient from doing the same on the backward pass (Xavier).

[^1]: `fan_in`/`fan_out`: the number of incoming and outgoing connections to a
  neuron, respectively.

The precise result of these functions applied on the model used here can be
seen in figure TODO, where the probability density functions for each
initialization method per layer are plotted.

![Plots for the probability density functions for each initialization method,
per layer.](./images/pdfs.png)

In addition to these four commonly used initialization methods, four extra
algorithms were tested, which can be obtained simply by halving or doubling the
outcome of both the Xavier equations, and plugging these values into the the
uniform or normal distribution in their stead. This effectively widens (for
doubling) or narrows (for halving) the resulting distributions compared to the
original version.

__Iterative, magnitude-based pruning.__ In general, pruning is the idea of
eliminating weights from a neural network, producing a smaller network.
Magnitude-based pruning specifies that the weights that will be nullified are
those in the top $X$ percent of weights closest to $0$ -- the intuition being
that the weights that are smallest in absolute terms are the least relevant to
the network's outcome and should be the first to go. Iterative, magnitude-based
pruning then describes a process where not all pruning is done in one step, but
over the course of multiple, spread-out pruning steps. [@frankle2019]

__Training.__ For all experiments, the network was trained for 100 epochs at a
learning rate of $0.0012$, during which it was pruned by $20\%$ and
subsequently reset to its initial weights every 5 epochs. 

![The progression of the size of the network over the course of one
experiment.](./images/pruning-progression.png)

__Testing.__ To reduce noise in the results from the experiments, each of the
eight initialization methods were run 11 times. The primary metric to inform
interpretation of the results is the mean test accuracy from these runs at
every fifth epoch -- right before pruning and resetting occurs --, with the
standard deviation over the runs acting as a metric of stability of this
resulting mean outcome.
