package test;

import java.awt.Color;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SequentialNN;
import utils.Activation;
import utils.UtilMethods;

import static utils.TensorUtils.*;

public class Test2{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(3, Activation.sigmoid));
		net.add(new FCLayer(4, Activation.softmax));
		// load the weights from a file
		net.loadFromFile("model.nn");
		
		Color[] intToColor1 = {Color.blue, Color.red, Color.yellow, Color.green};
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, null, null, null, (x2, y2) -> {
			return intToColor1[UtilMethods.argMax(net.predict(t(x2, y2)))];
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
