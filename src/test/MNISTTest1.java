package test;

import java.time.Duration;
import java.time.Instant;

import layer.FCLayer;
import network.SimpleNeuralNetwork;
import optimizer.*;
import utils.Activation;
import utils.Loss;
import utils.MNISTUtils;
import utils.UtilMethods;

public class MNISTTest1{
	public static void main(String[] args){
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(784);
		nn.add(new FCLayer(300, Activation.sigmoid));
		nn.add(new FCLayer(10, Activation.softmax));
		Instant start = Instant.now();
		double[][] x = MNISTUtils.loadDataSetImages("train-images-idx3-ubyte", Integer.MAX_VALUE);
		double[][] y = MNISTUtils.loadDataSetLabels("train-labels-idx1-ubyte", Integer.MAX_VALUE);
		nn.fit(x, y, 100, 50, Loss.crossEntropy, new AdamOptimizer(0.0001), 0.0, true);
		System.out.println("Training time: " + Duration.between(start, Instant.now()).toString());
		nn.saveToFile("mnist_weights.nn");
		double[][] testX = MNISTUtils.loadDataSetImages("t10k-images-idx3-ubyte", Integer.MAX_VALUE);
		double[][] testY = MNISTUtils.loadDataSetLabels("t10k-labels-idx1-ubyte", Integer.MAX_VALUE);
		double[][] testResult = nn.predict(testX);
		System.out.println("Classification accuracy: " + UtilMethods.format(UtilMethods.classificationAccuracy(testResult, testY)));
		MNISTTest2.main(args);
	}
}
