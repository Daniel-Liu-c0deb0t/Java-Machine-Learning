package test;

import java.awt.Color;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SimpleNeuralNetwork;
import optimizer.SGDOptimizer;
import utils.Activation;
import utils.Loss;
import utils.UtilMethods;

public class GraphTest1{
	public static void main(String[] args){
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2);
		net.add(new FCLayer(3, Activation.sigmoid));
		net.add(new FCLayer(4, Activation.softmax));
		
		double[][] x = UtilMethods.concat(UtilMethods.concat(UtilMethods.standardDist(0, 0, 0.1, 100), UtilMethods.standardDist(0, 1, 0.1, 100)), UtilMethods.concat(UtilMethods.standardDist(1, 0, 0.1, 100), UtilMethods.standardDist(1, 1, 0.1, 100)));
		double[][] y1 = new double[100][4];
		for(int i = 0; i < y1.length; i++){
			y1[i] = UtilMethods.oneHotEncode(0, 4);
		}
		double[][] y2 = new double[100][4];
		for(int i = 0; i < y2.length; i++){
			y2[i] = UtilMethods.oneHotEncode(1, 4);
		}
		double[][] y3 = new double[100][4];
		for(int i = 0; i < y3.length; i++){
			y3[i] = UtilMethods.oneHotEncode(2, 4);
		}
		double[][] y4 = new double[100][4];
		for(int i = 0; i < y4.length; i++){
			y4[i] = UtilMethods.oneHotEncode(3, 4);
		}
		double[][] y = UtilMethods.concat(UtilMethods.concat(y1, y2), UtilMethods.concat(y3, y4));
		net.fit(x, y, 1000, 4, Loss.crossEntropy, new SGDOptimizer(0.01), 0.01, true);
		
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
		
		Graph graph = new Graph(1000, 1000, xData, yData, cData, (x5, y5) -> {
			return intToColor1[UtilMethods.argMax(net.predict(new double[]{x5, y5}))];
		});
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("classification_example3.png", "png");
		graph.dispose();
		
		net.saveToFile("model2.nn");
	}
}
