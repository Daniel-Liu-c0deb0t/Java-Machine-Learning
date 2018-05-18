package tests;

import static javamachinelearning.utils.TensorUtils.t;

import java.awt.Color;

import javax.swing.JFrame;

import javamachinelearning.graphs.Graph;
import javamachinelearning.graphs.GraphPanel;
import javamachinelearning.layers.FCLayer;
import javamachinelearning.layers.ParamsLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.SGDOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;

public class ErrorGraphSquared{
	public static void main(String[] args){
		SequentialNN nn = new SequentialNN(1);
		ParamsLayer layer = new FCLayer(1, Activation.linear).noBias();
		nn.add(layer);
		
		Tensor[] x = {
				t(0),
				t(1),
				t(2),
				t(3),
				t(4)
		};
		
		Tensor[] y = {
				t(0),
				t(5),
				t(10),
				t(15),
				t(20)
		};
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, "Weight", "Error");
		
		double rangeStart = 3;
		double rangeEnd = 7;
		
		graph.useCustomScale(rangeStart, rangeEnd, 0, 50);
		
		int n = 20;
		double[] xs = new double[n];
		double[] ys = new double[n];
		for(int i = 0; i < n; i++){
			double xx = i * (rangeEnd - rangeStart) / n + rangeStart;
			xs[i] = xx;
			layer.setWeights(t(xx).reshape(1, 1));
			// y = 5x
			ys[i] = Loss.squared.loss(nn.predict(t(5)), t(5 * 5));
		}
		graph.addLineGraph(xs, ys);
		
		graph.draw();
		
		layer.setWeights(t(0.01).reshape(1, 1));
		
		nn.fit(x, y, 100, 1, Loss.squared, new SGDOptimizer(0.01), null, false, false, false, (epoch, loss) -> {
			graph.addPoint(layer.weights().flatGet(0),
					Loss.squared.loss(nn.predict(t(5)), t(5 * 5)), Color.green);
			graph.draw();
		});
		
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("error_graph_squared.png", "png");
	}
}
