package tests;

import javamachinelearning.layers.recurrent.GRUCell;
import javamachinelearning.layers.recurrent.RecurrentLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.AdamOptimizer;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;

public class GRUTest{
	public static void main(String[] args) throws Exception{
		SequentialNN nn = new SequentialNN(1, 2);
		nn.add(new RecurrentLayer(10, 2, new GRUCell()));
		nn.add(new RecurrentLayer(10, 1, new GRUCell()));
		
		Tensor[] x = {new Tensor(new double[][]{{0.1}, {0.2}})};
		Tensor[] t = {new Tensor(new double[][]{{0.2}, {0.3}})};
		
		nn.train(x, t, 1000, 1, Loss.squared, new AdamOptimizer(0.01), null, false, false, false);
		
		System.out.println(nn.predict(new Tensor(new double[][]{{0.1}, {0.2}})));
	}
}
