package test;

import java.awt.Color;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SequentialNN;
import optimizer.SGDOptimizer;
import utils.Activation;
import utils.Loss;
import utils.Tensor;
import utils.UtilMethods;

import static utils.TensorUtils.*;

public class GraphTest1{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(3, Activation.sigmoid));
		net.add(new FCLayer(4, Activation.softmax));
		
		Tensor[] x = UtilMethods.concat(UtilMethods.standardDist(0, 0, 0.1, 100), UtilMethods.standardDist(0, 1, 0.1, 100), UtilMethods.standardDist(1, 0, 0.1, 100), UtilMethods.standardDist(1, 1, 0.1, 100));
		Tensor[] y1 = new Tensor[100];
		for(int i = 0; i < y1.length; i++){
			y1[i] = UtilMethods.oneHotEncode(0, 4);
		}
		Tensor[] y2 = new Tensor[100];
		for(int i = 0; i < y2.length; i++){
			y2[i] = UtilMethods.oneHotEncode(1, 4);
		}
		Tensor[] y3 = new Tensor[100];
		for(int i = 0; i < y3.length; i++){
			y3[i] = UtilMethods.oneHotEncode(2, 4);
		}
		Tensor[] y4 = new Tensor[100];
		for(int i = 0; i < y4.length; i++){
			y4[i] = UtilMethods.oneHotEncode(3, 4);
		}
		Tensor[] y = UtilMethods.concat(y1, y2, y3, y4);
		net.fit(x, y, 100, 10, Loss.crossEntropy, new SGDOptimizer(1), 0.1, true, true, false);
		
		double[] xData = new double[x.length];
		double[] yData = new double[x.length];
		Color[] cData = new Color[x.length];
		Color[] intToColor1 = {Color.blue, Color.red, Color.yellow, Color.green};
		for(int i = 0; i < x.length; i++){
			xData[i] = x[i].flatGet(0);
			yData[i] = x[i].flatGet(1);
			cData[i] = intToColor1[UtilMethods.argMax(y[i])];
		}
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, xData, yData, cData, (x5, y5) -> {
			return intToColor1[UtilMethods.argMax(net.predict(t(x5, y5)))];
		});
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("classification_example3.png", "png");
		graph.dispose();
	}
}
