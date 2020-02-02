## Appendix A

### Theoretical definitions

__PyTorch__

The neural networks in this project are built using the open 
source machine learning package PyTorch. PyTorch is based on 
Python and is designed to make use of the tensor computing 
capabilities of GPU. Moreover, this framework includes modules 
for defining layers, saving model states, and forward- and 
backpropagation using an automatic differentiation module that 
records all operations that have been performed. These modules 
greatly reduce the amount of code and time required to build and 
train the networks, leaving more time for experiments and analysis 
of the results. By saving the state of the network before training 
has been performed, the configurations of the initial weights can 
later be analysed and compared against the performance of the 
corresponding networks after training and pruning.

__Git Repository__

For this project a Git repository was used whilst working
on the neural network. The repository can track changes made to 
files in a project. It builds a history of the project where all
these changes are saved.

__Trade off Sparsity and Accuracy__

Two key words that were of great importance in this project are 
sparsity and accuracy.
Sparsity is a measure of how many values in a certain weights 
group are 0. So, the higher the sparsity, the lower the amount 
of weights remaining. A high level of sparsity results in a 
smaller and faster neural network. But this might be at the 
cost of the accuracy of the neural network. The accuracy of a 
neural network is a measure of how well the network performs 
on given data after training. The higher the accuracy, the better 
the network perform its task, in this case, the better it 
recognizes written digits or pictures.

When implementing pruning, it is important to keep the trade-off 
between sparsity and accuracy in mind. It is advantagous to have 
a small and fast network, but it depending on the research it might 
be more important to keep a high accuracy. High sparsity can result 
in low accuracy, so it is essential to keep this balance in mind 
when doing research.

__Seeds__

Because of the pseudorandom nature of initialisation schemas a seed 
must be selected in order to sample the initial weights from the 
distributions specific to the schema used. Usually, a random seed is 
selected for this purpose. When the goal is to analyse the effects of 
design changes such as adjusting hyper-parameter values or, in the 
case of this project, different initialisation methods, keeping the 
seed constant between experiments removes the variation due to the 
randomness in seed selection. The resulting configuration of starting 
weights will be constant from one experiment to the next, thus making 
reproducibility of the results possible. To confirm that effects of 
deliberate adjustments were due to a structural difference between 
initialisation methods, as opposed to a result of the serendipity of 
the underlying random generation function, all experiments were rerun 
multiple times with different seeds. The results were then averaged over 
all runs, and the standard deviation between runs was calculated. When 
these means were eventually compared against other, their differences 
were be compared against the deviation between runs to determine the 
likelihood of this difference to be due to a relationship between the 
selected initialisation method and the results, rather than be due to chance.
