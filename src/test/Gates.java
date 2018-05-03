package test;

import layer.FCLayer;
import network.SimpleNeuralNetwork;
import optimizer.AdamOptimizer;
import utils.Activation;
import utils.Loss;
import utils.UtilMethods;

public class Gates{
	public static void main(String[] args){
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2);
		net.add(new FCLayer(2, Activation.sigmoid));
		net.add(new FCLayer(1, Activation.linear));
		double[][] x = {
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1}
		};
		double[][] y = {
				{0},
				{1},
				{1},
				{0}
		};
		net.fit(x, y, 3000, 4, Loss.squared, new AdamOptimizer(0.1), 0.0, true, true);
		
		UtilMethods.printArray(net.predict(new double[]{0, 0}));
		UtilMethods.printArray(net.predict(new double[]{1, 0}));
		UtilMethods.printArray(net.predict(new double[]{0, 1}));
		UtilMethods.printArray(net.predict(new double[]{1, 1}));
	}
}
