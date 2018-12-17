# Java Machine Learning Library
Java를 이용한 간단한 머신 러닝(신경망) 라이브러리입니다. 이 라이브러리는 교육을 목적으로 하고 있으며, 실제 프로젝트에서 사용하기에는 매우 느립니다.

(**주의: 구식코드입니다. 소스를 컴파일하세요.**)컴파일된 `.jar`파일을 다운로드 하고 당신의 프로젝트에 포함하고 싶으면 [여기](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/raw/master/JavaMachineLearning.jar)를 누르세요.

이 라이브러리는 최근에 많은 버그들을 고치고 내장된 tensor 클래스로 다양한 기능들을 포함한 벡터화된 연산을 사용하는 점검을 했습니다.

## 특징
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

## 사용 지침
이 라이브러리에서 제공하는 API는 고상하고(제 생각엔) 매우 높은 수준입니다. 모든 네트워크는 `SequentialNN` 클래스로 초기화하여 만들 수 있습니다. 이 클래스는 레이어를 추가하고 완전한 네트워크를 만들어 주는 도구를 제공합니다. 이 클래스를 초기화 할 때, 당신은 매개변수로서 입력의 형태를 명시해야 합니다. 

`SequentialNN`에 있는 `add` 메소드를 사용하여, 당신은 순차 모델(Sequential model)에 레이어를 추가할 수 있습니다. 이 레이어들은 순전파(forward propagation) 동안에 추가된 순서대로 평가됩니다. 순전파를 하려면, 예측 함수를 사용하고 tensor로 입력(들)을 제공합니다. Tensors는 내부적으로 열 주요 순서인 평면을 나타내는 다차원 배열입니다. 하지만, Tensors는 (정규) 행 주요 배열을 받는 일부 생성자를 제공합니다. 모델을 학습시키기 위해서, 입력 및 예측되는 목표 출력과 함께 `train` 메소드를 호출하세요. 이 메소드는 손실 함수(loss function), optimizer, regularizer 등과 같은 변화할 수 있는 많은 매개변수를 가지고 있습니다. callback 함수는 모든 학습 시기에도 제공될 수 있습니다.

`RecurrentLayer` 클래스를 추가하여, 입력과 출력을 여러 시간 단계로 걸칠 수 있습니다. 예를 들어, 순환 레이어(recurrent layer) 후에 전결합 레이어(fully connected layer)를 사용할 때, 전결합 레이어는 각 시간 단계 전체에 출력이 적용됩니다. 다른 추가점은 사용자가 지정한 시간 단계에 평가할 수 있게 하는 유연한 `predict` 함수입니다. 또한 순환 레이어는 다중 훈련 예제 또는 예측을 통해 상태저장을 할 수 있습니다.

### 바닐라 신경망

신경망을 사용한 간단한 선형 회귀(linear regression)를 실행하는 것이 얼마나 쉬운지를 보여주는 코드가 있습니다:
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
전체 소스 파일은 [여기](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/LinearGraph.java)에서 찾을 수 있습니다. `t` 메소드는 1차원 텐서를 만들기 위한 편리한 메소드일 뿐임을 주의해주세요. 전체 코드는 점과 선으로 이루어진 가중치(weight)/편향(bias) 그래프 창을 만듭니다:
![linear regression graph](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/nn_linear_regression.png)

약간 다른 데이터 집합(y = 5x + 3 대신 y = 5x, 노이즈가 없고, 편향치가 없음)에서, 가중치에 관한 손실/오류를 그래프로 그릴 수 있습니다:
![error wrt weight graph](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/error_graph_squared.png)
초록 점들은 훈련 알고리즘이 훈련을 통해 "찾아간" 가중치를 표시합니다. 손실 함수의 제곱으로 인해 이차식 그래프로 나타납니다. 그래프는 손실이 가장 적을 때인 최솟값으로 수렴되는 점에 주목해주세요. 이 최솟값은 우리가 학습하고자 하는 선형 함수의 기울기인 x = 5 중심에 있습니다.

다음 코드는 MNIST 필기 숫자 분류를 위해 3중 레이어 신경망을 훈련 하기 위한 것입니다.
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
전체 소스 파일은 [여기](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTFullyConnected.java)에서 찾을 수 있습니다.

### 콘볼루션 신경망

동일한 숫자 분류 작업을 위한 콘볼루션 레이어를 사용한 훈련 코드는 [here](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTConv.java)에서 찾을 수 있습니다. 하지만, 이 코드는 매우 느려서 모델이 직접적으로 일부의 숫자를 기억할 수 있는지를 보는 더 간단한 테스트를 수행했습니다. 그 코드는 [여기](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/TrainMNISTConvMemorize.java)에서 이용할 수 있습니다. 그 구조는 이전 콘볼루션 망과 매우 유사합니다:
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
이 망을 훈련하는 데는 20분 쯤 걸리고 입력 이미지 클래스를 완벽히 기억할 수 있습니다.

### 순환 신경망

순환 신경망을 만드는 것 또한 매우 간단합니다. 현재, GRU cells만 지원되고, 저는 일부 셰익스피어와 이상한 나라의 앨리스 글을 학습하고 생성하기 위해 그 것을 사용했습니다..

여기엔 하이퍼 파라미터가 사용되었습니다:
```java
int epochs = 500;
int batchSize = 10;
int winSize = 20;
int winStep = 20; // winSize = winStep so substrings are not repeated
int genIter = 5000; // how many characters to generate
double temperature = 0.1; // lower = less randomness
```
그리고 여기 2중 레이어 순환 신경망 모델을 빌드하는 코드가 있습니다.
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
모델을 훈련하고 본문을 생성하는 전체 코드를 원하면 [여기](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/blob/master/src/tests/GRUTest.java)로 오세요.

셰익스피어의 소네트 130번을 수행한 출력이 있습니다:
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
맨 처음 괄호 안에 있는 글은 제가 입력한 시드 텍스트입니다. 네트워크는 그 것을 받고 제가 지운 끝 부분에 빈 공간을 조금 더해서 소네트의 나머지 부분을 생성합니다.

여기 이상한 나라의 앨리스에서 발췌한 문단을 수행한 출력이 있습니다:
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
문단에는 철자가 틀린 단어가 매우 많지만, 꽤 멋져 보입니다.

마지막으로, 로미오와 줄리엣 1막 1장 전문으로 네트워크를 수행한 출력입니다:
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
당신도 보다시피 temperature가 증가(랜덤하게 증가)하면 아마도 사라질 반복이 있습니다. 원래, 저는 글이 무슨 막, 무슨 장인지 말해주는 첫 라인부터 예측을 시작하는 네트워크를 원했지만 네트워크는 다른 부분에서 실행됩니다.

이 모든 예제들에서, 모델과 하이퍼파라미터는 같습니다. 멋진 점은 이 네트워크는 텍스트의 구조를 학습하고 적절하게 새로운 줄과 들여쓰기를 첫 번째와 세 번째 예제에서 추가했다는 점입니다. 또한, 저는 이 글들을 [Project Gutenberg](http://www.gutenberg.org/)에서 가져왔습니다.

많은 다른 예제들은 [tests folder](https://github.com/Daniel-Liu-c0deb0t/Java-Machine-Learning/tree/master/src/tests)에서 찾을 수 있습니다.

저는 후전파(back propagation)와 경사 하강 방정식(gradient descent equations) 블로그 [게시글](https://c0deb0t.wordpress.com/2018/06/17/the-math-for-gradient-descent-and-backpropagation/)을 가지고 있습니다. 거기엔 흥미로운 수학 내용이 있습니다!

### 이미지 로드

텐서를 이용해 이미지를 로드하고 싶으면, 다음 코드를 따라하세요.

```
ImageUtils imgUtils = new ImageUtils();
Tensor imgTensor = img.readColorImageToTensor(String path, boolean convertGray)
```

그리고 텐서 배열을 이용해 많은 이미지를 로드하고 싶으면, 다음 코드 또한 따라할 수 있습니다.

```
ImageUtils imgUtils = new ImagUtils()
Tensor[] imgTensorArray = public Tensor[] readImages(String folderPath, boolean convertGray)
```
