package tests;

import javamachinelearning.layers.feedforward.ActivationLayer;
import javamachinelearning.layers.feedforward.ConvLayer;
import javamachinelearning.layers.feedforward.DropoutLayer;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.layers.feedforward.FlattenLayer;
import javamachinelearning.layers.feedforward.MaxPoolingLayer;
import javamachinelearning.layers.feedforward.ConvLayer.PaddingType;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.AdamOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.MNISTUtils;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class TrainMNISTConvMemorize{
	public static void main(String[] args) throws Exception{
		// training on the full MNIST data set is way too slow
		// to verify that the convolutional layers work, it is tested to memorize MNIST images
		
		// builds a convolutional neural network
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
		
		// loads the training data (only the first 100 images)
		Tensor[] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", 100);
		Tensor[] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", 100);
		
		long start = System.currentTimeMillis();
		
		nn.train(Utils.reshapeAll(x, 28, 28, 1),
				y,
				20, // number of epochs
				10, // batch size
				Loss.softmaxCrossEntropy,
				new AdamOptimizer(0.001),
				null, // no regularization
				true, // shuffle
				false);
		
		System.out.println("Training time: " + Utils.formatElapsedTime(System.currentTimeMillis() - start));
		
		// test on the images that the network was trained on
		Tensor[] testResult = nn.predict(Utils.reshapeAll(x, 28, 28, 1));
		
		System.out.println("Memorization accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, y)));
	}
}
