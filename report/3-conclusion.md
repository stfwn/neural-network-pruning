## Conclusion

Drawing on the material laid out in the previous section we can inch closer to
three conclusions about initialization methods when looking for winning tickets
following the lottery ticket hypothesis. In these conclusions, performance is
defined as having better test accuracy for lower percentages of weights
remaining after pruning.

* Normal distributions and uniform ones have similar performance.
* Distributions that are narrower around $0$ outperform wider distributions.
* Xavier methods outperform Kaiming methods.

Although a fair number of experiments have been done per initialization method,
and the standard deviations are small enough to have an acceptable level of
confidence in the results, more research should be done before these rules of
thumb can be set in stone. This project has only examined the application of
one pruning method on one neural network model trained on one dataset. That
being said, some interpretation for why these results arise from the
experiments is in order.

The reason for why normal and uniform distributions perform similarly could be
that the network is simply big enough to always give rise to the precise
composition of weights that make up the well-performing (although perhaps not
winning ticket-level) subnet. In other words: the sample size is so large that
the minor difference in the odds in the bell area of a normal distribution
versus the odds between the bounds under a uniform distribution simply does not
matter enough.

An explanation for why distributions that are narrower around $0$ outperform
wider distributions could lie in the fact that magnitude-based pruning is used
here. Perhaps if the weight initializations are closer together to begin with,
the network is more robust to pruning as the other weights can more easily be
optimized to fill the gap that was left. This is an interesting notion that
should be researched further by experimenting with different pruning methods
in combination with narrow and wide initialization methods.

Finally, the reason Xavier initialization methods outperform Kaiming methods
here should lie in the size of the weights in each layer relative to weights in
other layers. To clarify: notice that although we established narrower
distributions outperform wider ones by explicitly testing this factor in
isolation, the probability density functions that arise from Xavier methods are
in fact wider for every layer than those originating from Kaiming equations.
Then notice that _how much wider_ Xavier probability density functions are
differs per layer. This relationship between weight distributions from layer to
layer is an interesting pattern, and there may be a regularity here that can be
uncovered by doing further research.
