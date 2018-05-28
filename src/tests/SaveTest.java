package tests;

import static javamachinelearning.utils.TensorUtils.*;

import java.awt.Color;

import javax.swing.JFrame;

import javamachinelearning.graphs.Graph;
import javamachinelearning.graphs.GraphPanel;
import javamachinelearning.layers.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.SGDOptimizer;
import javamachinelearning.regularizers.L2Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;

public class SaveTest{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(3, Activation.relu));
		net.add(new FCLayer(4, Activation.softmax));
		
		Tensor[] x = {
				t(0, 0),
				t(0, 1),
				t(1, 0),
				t(1, 1),
				t(0.1, 0.1),
				t(0.1, 0.9),
				t(0.9, 0.1),
				t(0.9, 0.9)
		};
		
		Tensor[] y = {
				t(1, 0, 0, 0),
				t(0, 1, 0, 0),
				t(0, 0, 1, 0),
				t(0, 0, 0, 1),
				t(1, 0, 0, 0),
				t(0, 1, 0, 0),
				t(0, 0, 1, 0),
				t(0, 0, 0, 1)
		};
		
		net.fit(x, y, 1000, 4, Loss.softmaxCrossEntropy, new SGDOptimizer(0.1), new L2Regularizer(0.1), true, true, true);
		
		double[] xData = new double[x.length];
		double[] yData = new double[x.length];
		Color[] cData = new Color[x.length];
		Color[] intToColor1 = {Color.blue, Color.red, Color.yellow, Color.green};
		for(int i = 0; i < x.length; i++){
			xData[i] = x[i].flatGet(0);
			yData[i] = x[i].flatGet(1);
			cData[i] = intToColor1[argMax(y[i])];
		}
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, xData, yData, cData, (x2, y2) -> {
			return intToColor1[argMax(net.predict(t(x2, y2)))];
		});
		graph.draw();
		frame.add(new GraphPanel(graph));
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
		
		graph.saveToFile("classification_example.png", "png");
		graph.dispose();
		
		net.saveToFile("saved_model_test.nn");
	}
}
