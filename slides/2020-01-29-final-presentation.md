---
title: One Initialization to Rule Them All?
author: Stefan Wijnja, Ellis Wierstra, Thomas Hamburger
date: \today{}
---
# Neural networks are very cool

\center ![](images/health.png){ height=100px }     ![](images/stocks.jpeg){ height=100px }


\center ![](images/go.jpg){ height=100px }     ![](images/hurricane.jpg){ height=100px }

---

# But they can be very demanding

* Largest networks have billions of parameters
* Large clusters of GPUs needed to train
  - Limits accessibility
  - Limits applicability


---

__The Lottery Ticket Hypothesis__

A randomly initialized, dense neural-network contains a subnetwork that is
initalized such that -- when trained in isolation -- it can match the test
accuracy of the original network after training for at most the same number of
iterations.

[@frankle2019]

---

__The Lottery Ticket Hypothesis__ (redux)

A big net contains a small net that can match the big net.

---

# Finding winning tickets by pruning

* Pruning
* Magnitude-based pruning
* Iterative magnitude-based pruning

---

# Status quo on pruning

[@frankle2019]

* “Winning tickets” algorithm
  - 80-90% size reduction
  - Learn faster
  - Higher test accuracy

[@morcos2019]

* Finding winning tickets is computationally expensive
  - Solution: generate them once, reuse on other datasets
  - With very little performance loss

---

# Our question

\center
How do initialization algorithms compare when looking for winning tickets?

---

# Method
## Plan

1. Pick dataset.
2. Pick model.
3. Pick initialization methods.
4. __Write the code.__
5. __Do lots of experiments.__
6. Answer the question.

---

# Method
## Dataset: MNIST
\center
![](images/mnist.png){ height=150px }

* Handwritten digits
* $28 \times 28$ pixels
* Training set: 60k
* Test set: 10k

---

# Method
## Model: LeNet [@lecun1998]
\center
![](images/net.jpg){ height=150px }

* Fully connected
* Two hidden layers: 300 & 100 neurons $\rightarrow$ 266k weights.
* Leaky ReLU (negative slope: $0.05$)

---

# Method
## Initialization Methods

|                                |  | Xavier                                              | Kaiming                                    |
|--------------------------------|--|-----------------------------------------------------|--------------------------------------------|
| $\mathcal{U}(-a, a)$           |  | $\sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$ | $\sqrt{\frac{3}{\text{fan\_mode}}}$        |
| $\mathcal{N}(0, \text{std}^2)$ |  | $\sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$ | $\frac{\text{1}}{\sqrt{\text{fan\_mode}}}$ |

* `fan_in`/`fan_out`: number of inputs to/outputs from a neuron.
  `fan_mode`: `fan_in` or `fan_out`.

---

# Method

\center
Initialization Methods' Probability Density Functions

\center
![](images/pdfs.png){ width=95% }

---

# Method
## Training

1. Initialize network and save weights.
2. Train for 5 epochs.
3. Trim smallest 20% of the weights.
4. Reset the weights to the saved ones.
5. `GO TO 2`

\center
![](images/pruning-progression.png){ height=150px }

---

# Method
## Testing

* Save loss and accuracy on the training and testing set after every epoch.

\center
![](images/thinking-robot.png){ height=100px }

---


# Results

![](images/results-0-original-4.png)

---

![](images/results-1-xaviers.png)

---

\center
__Question__

\center
How do initialization algorithms compare when looking for winning tickets?


\center
__Answer__\*
\begin{align*}
\text{Xavier} &> \text{Kaiming}\\
\text{Uniform} &\approx \text{Normal}\\
\text{Narrow} &> \text{wide}
\end{align*}

---

# Further Research

* Grid search other datasets/models/pruning settings.
* Are pruning method and initialization method connected?

# Questions?

* Thanks
    * ![](images/andrei.jpg){ width=10px } Andrei Apostol
    * Putri van der Linden and Lieuwe Rekker
    * BrainCreators
* Stats
    * ~599 lines of code
    * 122 training runs
* Frameworks
    * PyTorch
* Can I reproduce your results?
    * Please do. Email s.wijnja@me.com

\center
![](images/braincreators.svg){ height=50px }

---

# References
