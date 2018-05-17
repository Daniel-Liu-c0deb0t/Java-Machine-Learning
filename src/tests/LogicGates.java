package tests;

import static javamachinelearning.utils.TensorUtils.t;

import javamachinelearning.layers.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.AdamOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;

public class LogicGates{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(2, Activation.sigmoid));
		net.add(new FCLayer(1, Activation.sigmoid));
		Tensor[] x = {
				t(0, 0),
				t(0, 1),
				t(1, 0),
				t(1, 1)
		};
		Tensor[] y = {
				t(0),
				t(1),
				t(1),
				t(0)
		};
		net.fit(x, y, 1000, 4, Loss.binaryCrossEntropy, new AdamOptimizer(0.1), null, true, true, true);
		
		System.out.println(net.predict(t(0, 0)));
		System.out.println(net.predict(t(1, 0)));
		System.out.println(net.predict(t(0, 1)));
		System.out.println(net.predict(t(1, 1)));
	}
}
