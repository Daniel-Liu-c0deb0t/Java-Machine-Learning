package test;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SimpleNeuralNetwork;
import optimizer.SGDOptimizer;
import utils.Loss;
import utils.UtilMethods;

public class GraphTest2{
	public static void main(String[] args){
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(1);
		nn.add(new FCLayer(1));
		
		double[][] x = {
				{0},
				{1},
				{2},
				{3},
				{4}
		};
		
		double[][] y = {
				{3 + 0 + 1},
				{3 + 5 - 1},
				{3 + 10 + 1},
				{3 + 15 - 1},
				{3 + 20 + 1}
		};
		
		nn.fit(x, y, 10, 1, Loss.squared, new SGDOptimizer(0.01), 0, true);
		
		UtilMethods.printArray(nn.predict(new double[]{5}));
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, UtilMethods.flattenData(x), UtilMethods.flattenData(y), null, null);
		graph.useCustomScale(0, 5, 0, 30);
		graph.addLine(nn.layers().get(0).edges()[0].getWeight(), nn.layers().get(0).getBias()[0]);
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("nn_linear_regression.png", "png");
	}
}
