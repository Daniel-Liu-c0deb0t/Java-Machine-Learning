package layer;

import edge.Edge;
import utils.Activation;

public interface Layer{
	public int nextSize();
	public int prevSize();
	public void init(int prevSize);
	public void init(int prevSize, double[][] weights, double[] bias);
	public double[] getBias();
	public Edge[] edges();
	public double[] forwardPropagate(double[] input);
	public Activation getActivation();
	public Activation getActivationP();
	public double getDropout();
}
