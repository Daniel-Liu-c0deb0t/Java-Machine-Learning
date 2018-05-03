package optimizer;

import edge.Edge;
import utils.Activation;

public interface Optimizer{
	public double optimizeWeight(double grad, int l, Edge e, int size, int max, int nextSize);
	public double optimizeBias(double grad, int l, int i, int size, int max, int nextSize);
}
