package test;

import layer.FCLayer;
import network.SimpleNeuralNetwork;
import optimizer.*;
import utils.Activation;
import utils.Loss;
import utils.MNISTDataSetLoader;
import utils.UtilMethods;

public class MNISTTest1{
	public static void main(String[] args){
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(784);
		nn.add(new FCLayer(300, Activation.sigmoid, Activation.sigmoidP));
		//nn.add(new FCLayer(100, Activation.sigmoid, Activation.sigmoidP));
		nn.add(new FCLayer(10, Activation.softmax, Activation.softmaxP));
		double[][] x = MNISTDataSetLoader.loadImages("train-images-idx3-ubyte", Integer.MAX_VALUE);
		double[][] y = MNISTDataSetLoader.loadLabels("train-labels-idx1-ubyte", Integer.MAX_VALUE);
		nn.fit(x, y, 100, 50, Loss.crossEntropyP, new AdamOptimizer(0.0002), 0, Loss.crossEntropy, true);
		nn.saveToFile("mnist_weights.nn");
		double[][] testX = MNISTDataSetLoader.loadImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		double[][] testY = MNISTDataSetLoader.loadLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		double[][] testResult = nn.predict(testX);
		System.out.println(UtilMethods.classificationAccuracy(testResult, testY));
	}
}
