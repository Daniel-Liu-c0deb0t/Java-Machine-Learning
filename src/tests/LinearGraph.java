package tests;

import static javamachinelearning.utils.TensorUtils.t;

import javax.swing.JFrame;

import javamachinelearning.graphs.Graph;
import javamachinelearning.graphs.GraphPanel;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.SGDOptimizer;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class LinearGraph{
	public static void main(String[] args){
		// neural network with 1 input and 1 output, no activation function
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
		
		nn.train(x,
				y,
				100, // number of epochs
				1, // batch size
				Loss.squared,
				new SGDOptimizer(0.01),
				null, // no regularizer
				false, //do not shuffle data
				true); // verbose
		
		// try the network on new data
		System.out.println(nn.predict(t(5)));
		
		JFrame frame = new JFrame();
		
		// graph the learned line
		Graph graph = new Graph(1000, 1000, Utils.flatCombine(x), Utils.flatCombine(y), null, null);
		graph.useCustomScale(0, 5, 0, 30);
		graph.addLine(((FCLayer)nn.layer(0)).weights().flatGet(0), ((FCLayer)nn.layer(0)).bias().flatGet(0));
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		// show the JFrame
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		// save the plot
		graph.saveToFile("nn_linear_regression.png", "png");
	}
}
