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
		SequentialNN nn = new SequentialNN(784);
		nn.add(new FCLayer(300));
		nn.add(new ActivationLayer(Activation.relu));
		nn.add(new FCLayer(10));
		nn.add(new ActivationLayer(Activation.softmax));
		
		Tensor[] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", Integer.MAX_VALUE);
		
		long start = System.currentTimeMillis();
		
		nn.train(Utils.flattenAll(x), y, 100, 100, Loss.softmaxCrossEntropy, new MomentumOptimizer(0.5, true), new L2Regularizer(0.0001), true, false);
		
		System.out.println("Training time: " + Utils.formatElapsedTime(System.currentTimeMillis() - start));
		
		nn.saveToFile("mnist_weights_fc.nn");
		
		Tensor[] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		Tensor[] testResult = nn.predict(Utils.flattenAll(testX));
		
		System.out.println("Classification accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, testY)));
	}
}
