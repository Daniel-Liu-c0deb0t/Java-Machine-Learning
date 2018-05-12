package test;

import layer.FCLayer;
import network.SequentialNN;
import optimizer.AdamOptimizer;
import utils.Activation;
import utils.Loss;
import utils.Tensor;

import static utils.TensorUtils.*;

public class LogicGates{
	public static void main(String[] args){
		SequentialNN net = new SequentialNN(2);
		net.add(new FCLayer(2, Activation.sigmoid));
		net.add(new FCLayer(1, Activation.linear));
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
		net.fit(x, y, 1000, 4, Loss.squared, new AdamOptimizer(0.1), 0.0, true, true, true);
		
		System.out.println(net.predict(t(0, 0)));
		System.out.println(net.predict(t(1, 0)));
		System.out.println(net.predict(t(0, 1)));
		System.out.println(net.predict(t(1, 1)));
	}
}
