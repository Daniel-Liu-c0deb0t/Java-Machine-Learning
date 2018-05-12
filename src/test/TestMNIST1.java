package test;

import layer.FCLayer;
import network.SequentialNN;
import utils.Activation;
import utils.MNISTUtils;
import utils.Tensor;
import utils.UtilMethods;

public class TestMNIST1{
	public static void main(String[] args){
		SequentialNN nn = new SequentialNN(784);
		nn.add(new FCLayer(300, Activation.sigmoid));
		nn.add(new FCLayer(10, Activation.softmax));
		nn.loadFromFile("mnist_weights.nn");
		
		Tensor[] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		Tensor[] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		Tensor[] testResult = nn.predict(UtilMethods.flattenAll(testX));
		
		System.out.println("Classification accuracy: " + UtilMethods.format(UtilMethods.classificationAccuracy(testResult, testY)));
	}
}
