package tests;

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

public class TrainMNISTConv{
	public static void main(String[] args) throws Exception{
		// very slow!
		
		SequentialNN nn = new SequentialNN(28, 28, 1);
		nn.add(new ConvLayer(5, 32, PaddingType.SAME, Activation.relu));
		nn.add(new MaxPoolingLayer(2, 2));
		nn.add(new ConvLayer(5, 64, PaddingType.SAME, Activation.relu));
		nn.add(new MaxPoolingLayer(2, 2));
		nn.add(new FlattenLayer());
		nn.add(new FCLayer(1024, Activation.relu));
		nn.add(new DropoutLayer(0.3));
		nn.add(new FCLayer(10, Activation.softmax));
		
		Tensor[] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", Integer.MAX_VALUE);
		
		long start = System.currentTimeMillis();
		
		nn.train(Utils.reshapeAll(x, 28, 28, 1), y, 100, 100, Loss.softmaxCrossEntropy, new AdamOptimizer(0.01), null, true, false, false);
		
		System.out.println("Training time: " + Utils.formatElapsedTime(System.currentTimeMillis() - start));
		
		nn.saveToFile("mnist_weights_conv.nn");
		
		Tensor[] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		Tensor[] testResult = nn.predict(Utils.reshapeAll(testX, 28, 28, 1));
		
		System.out.println("Classification accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, testY)));
	}
}
