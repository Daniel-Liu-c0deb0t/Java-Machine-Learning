package test;

import layer.FCLayer;
import network.SimpleNeuralNetwork;
import utils.Activation;
import utils.MNISTDataSetLoader;
import utils.UtilMethods;

public class MNISTTest2{
	public static void main(String[] args){
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(784);
		nn.add(new FCLayer(300, Activation.sigmoid, Activation.sigmoidP));
		nn.add(new FCLayer(10, Activation.softmax, Activation.softmaxP));
		nn.loadFromFile("mnist_weights.nn");
		double[][] testX = MNISTDataSetLoader.loadImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		double[][] testY = MNISTDataSetLoader.loadLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		double[][] testResult = nn.predict(testX);
		System.out.println("Classification accuracy: " + UtilMethods.classificationAccuracy(testResult, testY));
	}
}
