package tests;

import javamachinelearning.layers.feedforward.ActivationLayer;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.MomentumOptimizer;
import javamachinelearning.regularizers.L2Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.MNISTUtils;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class TrainMNISTFullyConnected{
	public static void main(String[] args){
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
	}
}
