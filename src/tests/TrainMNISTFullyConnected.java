package tests;

import javamachinelearning.layers.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.AdamOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.MNISTUtils;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class TrainMNISTFullyConnected{
	public static void main(String[] args){
		SequentialNN nn = new SequentialNN(784);
		nn.add(new FCLayer(300, Activation.relu));
		nn.add(new FCLayer(10, Activation.softmax));
		
		Tensor[] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", Integer.MAX_VALUE);
		
		long start = System.currentTimeMillis();
		
		nn.fit(Utils.flattenAll(x), y, 100, 32, Loss.softmaxCrossEntropy, new AdamOptimizer(0.1), null, true, false, false);
		
		System.out.println("Training time: " + Utils.formatElapsedTime(System.currentTimeMillis() - start));
		
		nn.saveToFile("mnist_weights.nn");
		
		Tensor[] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		Tensor[] testResult = nn.predict(Utils.flattenAll(testX));
		
		System.out.println("Classification accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, testY)));
	}
}
