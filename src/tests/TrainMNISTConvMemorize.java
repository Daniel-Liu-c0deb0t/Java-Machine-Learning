package tests;

import javamachinelearning.layers.ConvLayer;
import javamachinelearning.layers.DropoutLayer;
import javamachinelearning.layers.FCLayer;
import javamachinelearning.layers.FlattenLayer;
import javamachinelearning.layers.MaxPoolingLayer;
import javamachinelearning.layers.ConvLayer.PaddingType;
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
		
		SequentialNN nn = new SequentialNN(28, 28, 1);
		nn.add(new ConvLayer(5, 32, PaddingType.SAME, Activation.relu));
		nn.add(new MaxPoolingLayer(2, 2));
		nn.add(new ConvLayer(5, 64, PaddingType.SAME, Activation.relu));
		nn.add(new MaxPoolingLayer(2, 2));
		nn.add(new FlattenLayer());
		nn.add(new FCLayer(1024, Activation.relu));
		nn.add(new DropoutLayer(0.3));
		nn.add(new FCLayer(10, Activation.softmax));
		
		Tensor[] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", 100);
		Tensor[] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", 100);
		
		long start = System.currentTimeMillis();
		
		nn.fit(Utils.reshapeAll(x, 28, 28, 1), y, 20, 10, Loss.softmaxCrossEntropy, new AdamOptimizer(0.001), null, true, false, false);
		
		System.out.println("Training time: " + Utils.formatElapsedTime(System.currentTimeMillis() - start));
		
		Tensor[] testResult = nn.predict(Utils.reshapeAll(x, 28, 28, 1));
		
		System.out.println("Memorization accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, y)));
	}
}
