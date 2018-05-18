# Java-Machine-Learning
Simple machine learning library for Java. The library is mainly for educational purposes, and it is way too slow to be used on actual projects.

## Features:
- Feed-forward and backpropagration for different layers
  - Fully connected
  - Convolutional (2D convolution on 3D inputs with 4D weights)
  - Max/Average Pooling
  - Dropout
  - Flatten (Conv/Pool -> FC)
- Adam, Adagrad, and SGD optimizers
- Mini-batch gradient descent
  - Average weight adjustments through a batch
- Sigmoid, tanh, relu, and softmax activation functions
- L1, L2, and elastic net regularization
- Squared error, binary cross entropy, and multi-class cross entropy
  - Squared error for regression
  - Binary cross entropy + sigmoid activation for binary classification
  - Multi-class cross entropy + softmax activation for general classification
- Internally uses "tensors", which are multidimensional arrays/matrices
- Simple graphing class for graphing classification boundaries, points, lines, line plots, etc.
- MNIST dataset loader
- Save/load weights to/from files
- Drawing GUI for MNIST
- A bunch of testing classes and graphing examples

## Tutorial
The API provided by this library is quite elegant (in my opinion) and very high level. A whole network can be created by initializing a `SequentialNN` class. That class provides the tools to add layers and build a complete network. When initializing that class, you need to specify the shape of the input as the parameter.

Using the `add` method in `SequentialNN`, you can add layers to the sequential model. These layers will be evaluated in the order they are added during forward propagation. To forward propagate, use the predict function and provide input(s) as tensors. Tensors are multidimensional arrays that are represented in a flat, column major order format internally. However, provides a few contructors that accept (regular) row major arrays. To train a model, call the `fit` method. That method has many parameters that are used to train the model. A callback function can even be provided for every epoch of training. Note that 
