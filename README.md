# riderml
I heard somewhere that true understanding in computer science is equal parts intuition, math, and implementation.  This project contains machine learning algorithms that I have implemented for fun and my own education.

# Organization
The algorithms are split into categories:
 - neural_network contains neural network code
 - regression contains gradient descent algorithms
 - visualization contains code for visualizing examples for each algorithm
 - tests contains unit tests

# What ML algorithms are there now?
Currently the project contains feed-forward neural networks and gradient descent.  The neural network implementation is written in such a way that it should be easily extensible.  For example, each layer takes a series of functions for forward and backward propagation and so on, that could easily be swapped out.

nn.py
 - a generic layer model and the functions to use it as a component of a neural network.
 - backprop with irprop-
autoencoder.py
 - generic autoencoder with optional normalization for sparse denoising autoencoders.
dbn.py
 - deep belief networks -- arbitrary number of layers with greedy autoencoder pretraining.
gradient_descent.py
 - iRPROP- gradient descent with mini-batches or full data set
 - adagrad

# Visualization
Presently there are two approaches to visualization present in the repo. I originally used matplotlib but I have been increasingly using Bokeh.
![SGD]("https://raw.github.com/arider/riderml/master/images/gradient_descent.png")


# Requirements
You'll need scipy, numpy, matplotlib, and bokeh.

# Support
I have a bunch of implementations of other algorithms and I will add them in no particular order and at no particular time....

If you would like to contribute, please go right ahead.
