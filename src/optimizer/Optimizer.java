package optimizer;

import network.NeuralNetwork;

public interface Optimizer{
	public Deltas optimize(NeuralNetwork nn, double[][] result, double[] error, double lambda, double weightSum);
}
