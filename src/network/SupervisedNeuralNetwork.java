package network;

import optimizer.Optimizer;
import utils.Loss;

public interface SupervisedNeuralNetwork{
	public void fit(double[][] input, double[][] target, boolean verbose);
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, boolean verbose);
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, boolean verbose);
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, double lambda, boolean verbose);
}
