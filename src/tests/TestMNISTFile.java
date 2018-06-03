package tests;

import javamachinelearning.layers.feedforward.ActivationLayer;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.MNISTUtils;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class TestMNISTFile{
	public static void main(String[] args){
		SequentialNN nn = new SequentialNN(784);
		nn.add(new FCLayer(300));
		nn.add(new ActivationLayer(Activation.relu));
		nn.add(new FCLayer(10));
		nn.add(new ActivationLayer(Activation.softmax));
		nn.loadFromFile("mnist_weights_fc.nn");
		
		Tensor[] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		Tensor[] testResult = nn.predict(Utils.flattenAll(testX));
		
		System.out.println("Classification accuracy: " + Utils.format(Utils.classificationAccuracy(testResult, testY)));
	}
}
