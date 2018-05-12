package network;

import optimizer.Optimizer;
import utils.Loss;
import utils.Tensor;

public interface SupervisedNeuralNetwork{
	public void fit(Tensor[] input, Tensor[] target, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, boolean shuffle, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, boolean shuffle, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, double regLambda, boolean shuffle, boolean verbose, boolean printNet);
}
