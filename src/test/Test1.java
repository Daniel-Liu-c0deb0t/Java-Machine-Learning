package test;

import java.awt.Color;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SimpleNeuralNetwork;
import optimizer.*;
import utils.Activation;
import utils.Loss;
import utils.UtilMethods;

public class Test1{
	public static void main(String[] args){
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2);
		net.add(new FCLayer(3, Activation.sigmoid));
		net.add(new FCLayer(4, Activation.softmax));
		
		double[][] x = {
				{0, 0},
				{0, 1},
				{1, 0},
				{1, 1},
				{0.1, 0.1},
				{0.1, 0.9},
				{0.9, 0.1},
				{0.9, 0.9}
		};
		
		double[][] y = {
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{0, 0, 0, 1},
				{1, 0, 0, 0},
				{0, 1, 0, 0},
				{0, 0, 1, 0},
				{0, 0, 0, 1}
		};
		
		//UtilMethods.printNN(net);
		//int result = UtilMethods.maxDecode(net.predict(new double[]{0}));
		//System.out.println(result);
		//UtilMethods.printArray(result);
		//System.out.println();
		
		net.fit(x, y, 1000, 4, Loss.crossEntropy, new SGDOptimizer(0.1), 0.01, true);
		
		double[] xData = new double[x.length];
		double[] yData = new double[x.length];
		Color[] cData = new Color[x.length];
		Color[] intToColor1 = {Color.blue, Color.red, Color.yellow, Color.green};
		for(int i = 0; i < x.length; i++){
			xData[i] = x[i][0];
			yData[i] = x[i][1];
			cData[i] = intToColor1[UtilMethods.argMax(y[i])];
		}
		
		JFrame frame = new JFrame();
		
		//Color[] intToColor2 = {new Color(0.5f, 0.5f, 1.0f), new Color(1.0f, 0.5f, 0.5f)};
		
		Graph graph = new Graph(1000, 1000, xData, yData, cData, (x2, y2) -> {
			return intToColor1[UtilMethods.argMax(net.predict(new double[]{x2, y2}))];
		});
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("classification_example.png", "png");
		graph.dispose();
		
		net.saveToFile("model.nn");
		
		//double[] result = net.predict(new double[]{1, 0});
		//System.out.println(result);
		//UtilMethods.printArray(result);
	}
}
