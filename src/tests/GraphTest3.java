package tests;

import static javamachinelearning.utils.TensorUtils.t;

import java.awt.Color;

import javax.swing.JFrame;

import javamachinelearning.graphs.Graph;
import javamachinelearning.graphs.GraphPanel;
import javamachinelearning.layers.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.AdamOptimizer;
import javamachinelearning.regularizers.L2Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class GraphTest3{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(4, Activation.sigmoid));
		net.add(new FCLayer(1, Activation.sigmoid));
		
		Tensor[] x = Utils.concat(Utils.standardDist(0, 0, 0.1, 100), Utils.standardDist(0, 0.5, 0.1, 100),
				Utils.standardDist(0.5, 0, 0.1, 100), Utils.standardDist(0.5, 0.5, 0.1, 100));
		
		Tensor[] y1 = new Tensor[100];
		for(int i = 0; i < y1.length; i++){
			y1[i] = t(0);
		}
		Tensor[] y2 = new Tensor[100];
		for(int i = 0; i < y2.length; i++){
			y2[i] = t(1);
		}
		Tensor[] y3 = new Tensor[100];
		for(int i = 0; i < y3.length; i++){
			y3[i] = t(1);
		}
		Tensor[] y4 = new Tensor[100];
		for(int i = 0; i < y4.length; i++){
			y4[i] = t(0);
		}
		Tensor[] y = Utils.concat(y1, y2, y3, y4);
		
		net.fit(x, y, 100, 10, Loss.binaryCrossEntropy, new AdamOptimizer(1), new L2Regularizer(0.01), true, true, false);
		
		double[] xData = new double[x.length];
		double[] yData = new double[x.length];
		Color[] cData = new Color[x.length];
		Color[] intToColor1 = {Color.blue, Color.red};
		for(int i = 0; i < x.length; i++){
			xData[i] = x[i].flatGet(0);
			yData[i] = x[i].flatGet(1);
			cData[i] = intToColor1[(int)y[i].flatGet(0)];
		}
		
		JFrame frame = new JFrame();
		
		Graph graph = new Graph(1000, 1000, xData, yData, cData, (x5, y5) -> {
			return intToColor1[(int)Math.round(net.predict(t(x5, y5)).flatGet(0))];
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
