package network;

import optimizer.Optimizer;
import utils.Loss;

public interface SupervisedNeuralNetwork{
	public void fit(double[][] input, double[][] target, boolean verbose);
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, boolean verbose);
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, Loss lossP, Optimizer optimizer, Loss loss, boolean verbose);
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, Loss lossP, Optimizer optimizer, double lambda, Loss loss, boolean verbose);
}
