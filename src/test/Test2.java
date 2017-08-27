package test;

import java.awt.Color;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SimpleNeuralNetwork;
import utils.Activation;
import utils.UtilMethods;

public class Test2{
	public static void main(String[] args){
		SimpleNeuralNetwork net = new SimpleNeuralNetwork(2);
		net.add(new FCLayer(3, Activation.sigmoid, Activation.sigmoidP));
		net.add(new FCLayer(4, Activation.softmax, Activation.softmaxP));
		net.loadFromFile("model.nn");
		
//		double[] xData = new double[x.length];
//		double[] yData = new double[x.length];
//		Color[] cData = new Color[x.length];
		Color[] intToColor1 = {Color.blue, Color.red, Color.yellow, Color.green};
//		for(int i = 0; i < x.length; i++){
//			xData[i] = x[i][0];
//			yData[i] = x[i][1];
//			cData[i] = intToColor1[UtilMethods.argMax(y[i])];
//		}
		
		JFrame frame = new JFrame();
		
		//Color[] intToColor2 = {new Color(0.5f, 0.5f, 1.0f), new Color(1.0f, 0.5f, 0.5f)};
		
		Graph graph = new Graph(1000, 1000, null, null, null, (x2, y2) -> {
			return intToColor1[UtilMethods.argMax(net.predict(new double[]{x2, y2}))];
		});
		graph.useCustomScale(0, 1, 0, 1);
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("classification_example2.png", "png");
		graph.dispose();
	}
}
