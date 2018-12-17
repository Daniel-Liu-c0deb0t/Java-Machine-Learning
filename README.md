# Java Machine Learning Library
Simple machine learning (neural network) library for Java. The library is mainly for educational purposes, and it is way too slow to be used on actual projects.

The Korean translation of this README.md is [here](README_ko.md), if you prefer to read it in Korean.

(**NOTE: PROBABLY OUTDATED. Just compile from source.**) If you want to download the compiled `.jar` file and include it to your own project, click [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/raw/master/JavaMachineLearning.jar).

This library recently got an overhaul that fixed many bugs and uses vectorized operations with a built-in tensor class, among many other features. The source code was also organized and comments were added.

## Features
- Feed-forward layers
  - Fully connected
  - Convolutional (2D convolution on 3D inputs with 4D weights)
  - Max/Average Pooling
  - Dropout
  - Activation
  - Flatten (Conv/Pooling -> FC)
  - Scaling
- Recurrent layer
  - GRU Cells
- Adam, Adagrad, momentum, NAG, Nesterov, SGD, RMSProp and AdaDelta optimizers
- Mini-batch gradient descent
  - Average gradients for each weight throughout each batch
- Sigmoid, tanh, relu, hard sigmoid, and softmax activation functions
- L1, L2, and elastic net regularization
- Squared loss, binary cross entropy, and multi-class cross entropy
  - Squared loss for regression
  - Binary cross entropy + sigmoid activation for binary classification
  - Multi-class cross entropy + softmax activation for general classification
- Internally uses "tensors", which are multidimensional arrays/matrices
- Simple graphing class for graphing classification boundaries, points, lines, line plots, etc.
- MNIST dataset loader
- Save/load weights to/from files
- Drawing GUI for MNIST
- A bunch of testing classes and graphing examples
- Image preprocessing

## Tutorial
The API provided by this library is quite elegant (in my opinion) and very high level. A whole network can be created by initializing a `SequentialNN` class. That class provides the tools to add layers and build a complete network. When initializing that class, you need to specify the shape of the input as the parameter.

Using the `add` method in `SequentialNN`, you can add layers to the sequential model. These layers will be evaluated in the order they are added during forward propagation. To forward propagate, use the predict function and provide input(s) as tensors. Tensors are multidimensional arrays that are represented in a flat, column major order format internally. However, it provides a few constructors that accept (regular) row major arrays. To train a model, call the `train` method with inputs and expected target outputs. This method has many parameters that can be changed, such as the loss function, optimizer, regularizer, etc. A callback function can even be provided for every epoch of training.

With the addition of a `RecurrentLayer` class, inputs and outputs can span many time steps. For example, when using fully connected layer after a recurrent layer, the fully connected layer is applied to the outputs for every single time step. Another addition is a flexible `predict` function that allows a custom number of time steps to be evaluated. Recurrent layers can also be stateful throughout multiple training examples or predictions.

### Vanilla Neural Networks

Here is a piece of code that shows how easy it is to run a simple linear regression using a neural network:
```java
// neural network with 1 input and 1 output, no activation function
SequentialNN nn = new SequentialNN(1);
nn.add(new FCLayer(1));

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

nn.train(x,
	y,
	100, // number of epochs
	1, // batch size
	Loss.squared,
	new SGDOptimizer(0.01),
	null, // no regularizer
	false, //do not shuffle data
	true); // verbose

// try the network on new data
System.out.println(nn.predict(t(5)));
```
You can find the full source file [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/LinearGraph.java). Note that the `t` method is just a convenience method to create 1D tensors. The full code will produce a window with the points and the line formed by the weight/bias graphed:
![linear regression graph](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/nn_linear_regression.png)

On a slightly different set of data (y = 5x instead of y = 5x + 3, no noise, and no bias), the loss/error with respect to the weight can be graphed:
![error wrt weight graph](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/error_graph_squared.png)
The green dots represent weights that the training algorithm "visited" throughout training. The quadratic shape of the graph is due to the squared loss function. Note that it converges to the minimum, where the loss is the lowest, and that minimum is centered on x = 5, which is the slope of the linear function that we want to learn.

The following piece of code is for training a 3 layer neural network for the MNIST handwritten digit classification.
```java
// create a model with 784 input neurons, 300 hidden neurons, and 10 output neurons
// use RELU for the hidden layer and softmax for the output layer
SequentialNN nn = new SequentialNN(784);
nn.add(new FCLayer(300));
nn.add(new ActivationLayer(Activation.relu));
nn.add(new FCLayer(10)); // 10 categories of numbers
nn.add(new ActivationLayer(Activation.softmax));

// load the training data
Tensor[] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", Integer.MAX_VALUE);
Tensor[] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", Integer.MAX_VALUE);

long start = System.currentTimeMillis();

nn.train(Utils.flattenAll(x),
	y,
	100, // number of epochs
	100, // batch size
	Loss.softmaxCrossEntropy,
	new MomentumOptimizer(0.5, true),
	new L2Regularizer(0.0001),
	true, // shuffle the data after every epoch
	false);

System.out.println("Training time: " + Utils.formatElapsedTime(System.currentTimeMillis() - start));

// save the learned weights
nn.saveToFile("mnist_weights_fc.nn");

// predict on previously unseen testing data
Tensor[] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
Tensor[] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
Tensor[] testResult = nn.predict(Utils.flattenAll(testX));

// prints the percent of images classified correctly
System.out.println("Classification accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, testY)));
```
The full code can be found [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTFullyConnected.java).

### Convolutional Neural Networks

The training code that uses convolutional layers for the same digit classification task can be found [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTConv.java). However, the code is very slow, so a simpler test to see if the model can directly memorize some digits was conducted. The code is available [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTConvMemorize.java). The architecture is very similar to the previous convolutional network:
```java
SequentialNN nn = new SequentialNN(28, 28, 1);

nn.add(new ConvLayer(5, 32, PaddingType.SAME));
nn.add(new ActivationLayer(Activation.relu));
nn.add(new MaxPoolingLayer(2, 2));

nn.add(new ConvLayer(5, 64, PaddingType.SAME));
nn.add(new ActivationLayer(Activation.relu));
nn.add(new MaxPoolingLayer(2, 2));

nn.add(new FlattenLayer());

nn.add(new FCLayer(1024));
nn.add(new ActivationLayer(Activation.relu));

nn.add(new DropoutLayer(0.3));

nn.add(new FCLayer(10));
nn.add(new ActivationLayer(Activation.softmax));
```
Training this network takes around 20 minutes and it can memorize the input image's classes perfectly.

### Recurrent Neural Networks

Creating a recurrent neural network is also very simple. Currently, only GRU cells are supported, and I used that to learn and generate some Shakespeare and Alice's Adventures in Wonderland text.

Here are the hyperparameters used:
```java
int epochs = 500;
int batchSize = 10;
int winSize = 20;
int winStep = 20; // winSize = winStep so substrings are not repeated
int genIter = 5000; // how many characters to generate
double temperature = 0.1; // lower = less randomness
```
And here is the code that builds the 2 layer recurrent neural network model:
```java
// for each time step, the input is a one hot vector describing the current character
// for each time step, the output is a one hot vector describing the next character
// the recurrent layers are stateful, which means that the next state relies on the previous states
SequentialNN nn = new SequentialNN(winSize, alphabet.length());
nn.add(new RecurrentLayer(winSize, new GRUCell(), true));
nn.add(new DropoutLayer(0.3));
nn.add(new RecurrentLayer(winSize, new GRUCell(), true));
// the same fully connected layer is applied for every single time step
nn.add(new FCLayer(alphabet.length()));
// scales the values by the temperature before softmax
nn.add(new ScalingLayer(1 / temperature, false));
nn.add(new ActivationLayer(Activation.softmax));
```
Go [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/GRUTest.java) if you want the full code for training the model and generating text.

Here is the output from running on Shakespear's Sonnet #130:
```
[cxx]x

  my mistress' eyes are nothing like the sun
  coral is far more red, than her lips red
  if snow be white, why then her breasts are dun
  if hairs be wires, black wires grow on her head.
  i have seen roses damask'd, red and white,
  but no such roses see i in her cheeks
  and in some perfumes is there more delight
  than in the breath that from my mistress reeks.
  i love to hear her speak, yet well i know
  that music hath a far more pleasing sound
  i grant i never saw a goddess go,--
  my mistress, when she walks, treads on the ground
    and yet by heaven, i think my love as rare,
    as any she belied with false compare
```
The text in brackets at the very beginning is the seed text entered in by me. The network takes that and generates the rest of the sonnet, plus some extra spaces at the end that I removed.

Here is the output from running on an excerpt of Alice's Adventures in Wonderland:
```
[chapter] i. down the rabbit-hole

alice was beginning then she
ray so menty see.

af the
hing howver be world she was considering in her feet, for it flashed across her mind that she ought to have wondered at the sides it pocts tow  th the tried to have wondered at the sides it pocts top the rabbit with pink eyes time as she fell very slowly, for she had
plenty of time as she went lothe the down nothing to her owa get in to her that she was considering in her feet, for it flashed across her mind that she ought to have wondered at the sides it withing either a waistcoat-pocket, and to ple pfonsidering in her feet, for it flashed across her mind there she fell very slowly, for she had
plenty so  it with pink ey her feet, for it flashed across her mind that she ought to have wondered at the sides it pocts top the rabbit with pink eyes time as she fell very slowly, for she had
plenty of time as she went tring to look down and make out what
she was considering in her feet, for it flashed across her mind that she ought to have wondered at the sides it poct plap the had pe was coming to, but it was too dark to see anything then she
looked at the sides all seemed quite was beginning then she
ray co peer a watch
to take out of it, and fortunately was just in time to see it pop down a large
rabbit-hole under the well was considering in her feet, for it flashed across her mind that she ought to have wondered at the sides it pocts top the rabbit with pink eyes time as she fell very slowly, for she had
plenty so  it with pink ey, she she to out it was too dark to tires all, be late! (when she thought it over afterwatd, but it was too dark to te was beginning then somenasy
seen a rabbit with either a waistcoat-pocket, and to ple ppend.n ahelves thok down a jar from one of the shelves had wondel very sleepy and stupid), whether the well was coused it was labelled orange maran
aling she to her feet, for it flashed across her mind there she fell very slowly, for she had
plenty of time as she went tr.e hed as
she pagllidy, nothing then she
ron to happen next. first, she tried to look down and make out what
she was considering in her feet, for it flashed across her mind that she ought to have wondered at the sides it poct poud there she fell past it
```
There are a lot of misspelled words, but it is pretty cool nonetheless.

Finally, here is the output of running the network on the entire Act I Scene I of Romeo and Juliet:
```
[act i]t sarn'd to the will part thee.

  rom. i do beauty his sunpong sprice.

  rom. i do beauty his groanse the wall.

  samp. i do beaut's thess swords therefere in that is to streponse thee the wall.

  samp. i do beaut's thess swords therefere if thou doth the maids, or ment to the willat them, an thet thee weart of lovers' from the strunce of his will the live his will be comes to the but whett ther theis ments i will the hat with the fair,
    bees thee, when the wall the hat with the wall.

  samp. i do beaut's thess swords therefere in that len.

  ben. montague should be so fair mark.
  sh therefore i will they will stoul of the maids having the will part thee the hat let pee i pass me not here in sparkling hath the maids hour side i sad the wall.

  samp. i do beaut's thess swords therefere in that len.

  ben. montague should more or here with the maids hours shown so thee with me.

  samp. i do beauty his caming makes the he will.

  samp. no, should montagues!
    wher thear hears'd and will they were head the willat to stand, and moved.

  ben. in sand hear the with my a wist.

  samp. i do beaut's thess sword morte the wall.

  samp. i do beaut's this will stans.

  greg. the heads of the beauty the hasterte me my fair madk to the was the weakes starn,
    where is to stor. what her comes is to store.

  samp. i do beauty the wall the hat here in sparkling his hit tles, and montague and with me.

  samp. i do betheas in paine.

  samp. a dog of the fair markman.

  samp. i do beaut's thess sword morte me they me what hes if othen. i will the wall.

  samp. i do beaut's thess swords therefere in that me.

  rom. i dis ment good the was she head the was what, and hent sword of the will part the will part the will.

  samp. no, sir.

  samp. no, as the weadt of the wall.

  samp. i do beaut's thess sword.

  rom. i dis ment good the wall.

  samp. a dog of the hat heads her his comes and montague ind sen the wall.

  samp. i do beaut's thess sword morte and made is that we dows thee i sang the hearty saive in strun.
```
As you can see there are some repetitions that would probably disappear if the temperature is increased (which increases the randomness). Originally, I wanted the network to start predicting from the first line that says what act and scene it was, but the network started from somewhere else.

In all of these examples, the model and hyperparameters were the same. What's cool is that the network learns the structure of the text and properly adds newlines and indents for the first and third examples. Also, I got the texts from [Project Gutenberg](http://www.gutenberg.org/).

Many other examples can be found in the [tests folder](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/tree/master/src/tests).

I have a blog post on backpropagation and gradient descent equations [here](https://c0deb0t.wordpress.com/2018/06/17/the-math-for-gradient-descent-and-backpropagation/). It has some interesting math stuff!

### Image load

If you want to load image to Tensor, you can do following codes

```
ImageUtils imgUtils = new ImageUtils();
Tensor imgTensor = img.readColorImageToTensor(String path, boolean convertGray)
```

And if you want to load many images to Tensor array, also you can do following codes

```
ImageUtils imgUtils = new ImagUtils()
Tensor[] imgTensorArray = public Tensor[] readImages(String folderPath, boolean convertGray)
```
