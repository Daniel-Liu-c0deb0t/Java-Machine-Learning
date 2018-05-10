package network;

import optimizer.Optimizer;
import utils.Loss;
import utils.Tensor;

public interface SupervisedNeuralNetwork{
	public void fit(Tensor[] input, Tensor[] target, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, double regLambda, boolean verbose, boolean printNet);
}
