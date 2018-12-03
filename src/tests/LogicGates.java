package tests;

import static javamachinelearning.utils.TensorUtils.t;

import javamachinelearning.layers.feedforward.ActivationLayer;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.MomentumOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;

public class LogicGates{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(2));
		net.add(new ActivationLayer(Activation.sigmoid));
		net.add(new FCLayer(1));
		net.add(new ActivationLayer(Activation.sigmoid));
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
		
		System.out.println(net);
		
		net.train(x, y, 2000, 4, Loss.binaryCrossEntropy, new MomentumOptimizer(0.1), null, true, true);
		
		System.out.println(net.predict(t(0, 0)));
		System.out.println(net.predict(t(1, 0)));
		System.out.println(net.predict(t(0, 1)));
		System.out.println(net.predict(t(1, 1)));
	}
}
