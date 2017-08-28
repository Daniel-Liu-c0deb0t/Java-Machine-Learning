package optimizer;

import edge.Edge;
import utils.Activation;

public interface Optimizer{
	public double optimizeWeight(int l, Edge e, double[] prevResult, double[] nextResult, double[] error, double lambda, double weightSum, Activation activation, int size, int max, int prevSize);
	public double optimizeBias(int l, int i, double[] nextResult, double[] error, Activation activation, int size, int max);
}
