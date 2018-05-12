package test;

import javax.swing.JFrame;

import graph.Graph;
import graph.GraphPanel;
import layer.FCLayer;
import network.SequentialNN;
import optimizer.SGDOptimizer;
import utils.Loss;
import utils.Tensor;
import utils.UtilMethods;

import static utils.TensorUtils.*;

public class GraphTest2{
	public static void main(String[] args){
		SequentialNN nn = new SequentialNN(1);
		nn.add(new FCLayer(1));
		
		// y = 5x + 3
		Tensor[] x = {
				t(0),
				t(1),
				t(2),
				t(3),
				t(4)
		};
		
		Tensor[] y = {
				t(3 + 0 + 1),
				t(3 + 5 - 1),
				t(3 + 10 + 1),
				t(3 + 15 - 1),
				t(3 + 20 + 1)
		};
		
		nn.fit(x, y, 100, 1, Loss.squared, new SGDOptimizer(0.01), 0, false, true, true);
		
		System.out.println(nn.predict(t(5)));
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, UtilMethods.flatCombine(x), UtilMethods.flatCombine(y), null, null);
		graph.useCustomScale(0, 5, 0, 30);
		graph.addLine(nn.layer(0).weights().flatGet(0), nn.layer(0).bias().flatGet(0));
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("nn_linear_regression.png", "png");
	}
}
