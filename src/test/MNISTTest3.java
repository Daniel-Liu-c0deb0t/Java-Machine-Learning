package test;

import layer.FCLayer;
import network.SimpleNeuralNetwork;
import utils.Activation;
import utils.MNISTUtils;
import utils.UtilMethods;

public class MNISTTest3{
	public static void main(String[] args){
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(784);
		nn.add(new FCLayer(300, Activation.sigmoid));
		nn.add(new FCLayer(10, Activation.softmax));
		nn.loadFromFile("mnist_weights.nn");
		
		double[] image = MNISTUtils.loadImage("test_mnist_image.jpg", 28, 28);
		UtilMethods.printImage(new double[][]{image});
		double[] result = nn.predict(image);
		System.out.println(UtilMethods.argMax(result));
	}
}
