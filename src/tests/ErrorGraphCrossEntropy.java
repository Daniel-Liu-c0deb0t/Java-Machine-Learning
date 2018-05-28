package tests;

import static javamachinelearning.utils.TensorUtils.t;

import java.awt.Color;

import javax.swing.JFrame;

import javamachinelearning.graphs.Graph;
import javamachinelearning.graphs.GraphPanel;
import javamachinelearning.layers.FCLayer;
import javamachinelearning.layers.FeedForwardParamsLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.SGDOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class ErrorGraphCrossEntropy{
	public static void main(String[] args) throws Exception{
		SequentialNN nn = new SequentialNN(1);
		FeedForwardParamsLayer layer = new FCLayer(1, Activation.sigmoid).noBias();
		nn.add(layer);
		
		Tensor[] x = Utils.concat(Utils.standardDist(-0.5, -0.5, 0.1, 100), Utils.standardDist(0.5, 0.5, 0.1, 100));
		x = Utils.reshapeAll(x, 1);
		
		Tensor[] y = new Tensor[x.length];
		
		int idx = 0;
		for(int i = 0; i < 100; i++){
			y[idx] = t(0);
			idx++;
		}
		for(int i = 0; i < 100; i++){
			y[idx] = t(1);
			idx++;
		}
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, "Weight", "Error");
		
		double rangeStart = 0;
		double rangeEnd = 10;
		
		graph.useCustomScale(rangeStart, rangeEnd, 0, 1);
		
		int n = 10;
		double[] xs = new double[n];
		double[] ys = new double[n];
		for(int i = 0; i < n; i++){
			double xx = i * (rangeEnd - rangeStart) / n + rangeStart;
			xs[i] = xx;
			layer.setWeights(t(xx).reshape(1, 1));
			ys[i] = Loss.binaryCrossEntropy.loss(nn.predict(t(0.3)), t(1));
		}
		graph.addLineGraph(xs, ys);
		
		graph.draw();
		
		layer.setWeights(t(0.01).reshape(1, 1));
		
		nn.fit(x, y, 100, 1, Loss.binaryCrossEntropy, new SGDOptimizer(1), null, false, false, false, (epoch, loss) -> {
			graph.addPoint(layer.weights().flatGet(0),
					Loss.binaryCrossEntropy.loss(nn.predict(t(0.3)), t(1)), Color.green);
			graph.draw();
		});
		
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("error_graph_cross_entropy.png", "png");
	}
}
