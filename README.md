# Java-Machine-Learning
Simple machine learning library for Java. The library is mainly for educational purposes, and it is way too slow to be used on actual projects.

This library recently got an overhaul that fixed many bugs and uses vectorized operations with a built-in tensor class, among many other features. The source code was also organized and comments were added.

## Features:
- Feed-forward and backpropagration for different layers
  - Fully connected
  - Convolutional (2D convolution on 3D inputs with 4D weights)
  - Max/Average Pooling
  - Dropout
  - Flatten (Conv/Pooling -> FC)
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

Using the `add` method in `SequentialNN`, you can add layers to the sequential model. These layers will be evaluated in the order they are added during forward propagation. To forward propagate, use the predict function and provide input(s) as tensors. Tensors are multidimensional arrays that are represented in a flat, column major order format internally. However, provides a few contructors that accept (regular) row major arrays. To train a model, call the `fit` method with inputs and expected target outputs. This method has many parameters that can be changed, such as the loss function, optimizer, regularizer, etc. A callback function can even be provided for every epoch of training.

Here is a piece of code that does linear regression and graphs the resulting line.
```java
SequentialNN nn = new SequentialNN(1);
nn.add(new FCLayer(1, Activation.linear));

// y = 5x + 3
Tensor[] x = {
	t(0),
	t(1),
	t(2),
	t(3),
	t(4)
};

Tensor[] y = {
	t(3 + 0 + 1),
	t(3 + 5 - 1),
	t(3 + 10 + 1),
	t(3 + 15 - 1),
	t(3 + 20 + 1)
};

nn.fit(x, y, 100, 1, Loss.squared, new SGDOptimizer(0.01), null, false, true, true);

System.out.println(nn.predict(t(5)));

JFrame frame = new JFrame();

Graph graph = new Graph(1000, 1000, Utils.flatCombine(x), Utils.flatCombine(y), null, null);
graph.useCustomScale(0, 5, 0, 30);
graph.addLine(((FCLayer)nn.layer(0)).weights().flatGet(0), ((FCLayer)nn.layer(0)).bias().flatGet(0));
graph.draw();
frame.add(new GraphPanel(graph));

frame.setSize(1200, 1200);
frame.setLocationRelativeTo(null);
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
frame.setVisible(true);

graph.saveToFile("nn_linear_regression.png", "png");
```
You can find the full source file [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/LinearGraph.java). Note that the `t` method is just a convenience method to create 1D tensors. This code will produce a window with the points and the line formed by the weight/bias graphed:
![linear regression graph](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/nn_linear_regression.png)

On a slightly different set of data (y = 5x instead of y = 5x + 3, no noise, and no bias), the error with respect to the weight can be graphed:
![error wrt weight graph](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/error_graph_squared.png)

The training code for MNIST with 3 fully connected layers can be found [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTFullyConnected.java).

The training code that uses conv, dropout, and fully connected layers can be found [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTConv.java).

However, the conv code is very slow, so a simpler test to see if the model can directly memorize some digits was conducted. The code is available [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTConvMemorize.java). The architecture is very similar to the previous conv network.

These and many other examples can be found in the [tests folder](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/tree/master/src/tests).
