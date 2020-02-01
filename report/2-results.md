## Results

Figure \ref{results-0} shows the test accuracy of each initialization method as
iterative pruning is applied over the course of the experiment. Up to the point
where only 3.5% of the weights remain, the mean performance of all four methods
is very similar to one another, and every method on its own shows stable
results in terms of deviation from its mean when running experiments
repeatedly.

It is immediately worth nothing that conform to @frankle2019's lottery ticket
hypothesis, pruned versions of the original network are matching and even
outperforming the test accuracy achieved initially. Winning tickets are being
found up to 6.9% weights remaining.

![\label{results-0}Each line shows the mean test accuracy from 11 differently seeded runs on one
initialization method, in order to produce a reliable plot. The error bars
represent the standard deviation between the runs; an indication for the
stability of the test accuracy at that point.  The displayed test accuracy was
measured every fifth epoch, right before pruning and
resetting.](images/results-0-original-4.png)

It could be argued this concludes the research, as we are technically not
finding winning tickets anymore beyond this level of sparsity -- all of the
networks' performances dropped below the accuracy of the original, full-size
network at around the 6.9% mark. This disquilifies them from winning the
lottery as defined by @frankle2019. However, further inspection uncover
regularities that may be useful for further research.

After the 3.5% mark, both Xavier-initalized networks perform better than both
Kaiming-initalized ones in the mean and in terms of stability around this mean.
Kaiming initalizations led to lower and more erratic performance. This pattern
does not hold for long. At 1.8% weights remaining, all the initialization
methods show too much internal instability to interpret their performance
meaningfully.

Xavier-uniform and Xavier-normal perform in much the same way through the
experiment, and the same holds for the Kaiming-normal and Kaiming-uniform pair.
Thus, thether the initialization originated from a uniform or normal
distribution did not make a difference in these experiments -- the deciding
difference was the formula that decided the parameters of these distributions.

Further experiments with the Xavier method of initialization have led to the
results shown in figure \ref{results-1}. Here both variants of the original
Xavier distributions -- normal and uniform -- have been narrowed or widened by
a factor of 2.

![\label{results-1}This plot shows the same type of information as in figure
\ref{results-0}, but for Xavier distributions that have been widened
(`*-double`) or narrowed (`*-half`) by a factor of
2.](images/results-1-xaviers.png)

The numbers show that both narrowed distributions outperform the other
initialization strategies at 2.3% weights remaining in the mean and with a
smaller standard deviation -- i.e.: they are more stable in their outcome over
multiple experiments --, but immediately afterwards the same erratic behavior
as appeared in the previous setting re-emerges.
