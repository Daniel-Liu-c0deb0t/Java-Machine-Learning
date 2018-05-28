package javamachinelearning.networks;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;

public interface SupervisedNeuralNetwork{
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle, boolean verbose, boolean printNet);
	public void fit(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle, boolean verbose, boolean printNet, ProgressFunction f);
	
	// callback function to check the progress of training
	public interface ProgressFunction{
		public void apply(int epoch, double loss);
	}
}
