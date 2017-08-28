package layer;

import java.nio.ByteBuffer;

import edge.Edge;
import optimizer.Optimizer;
import optimizer.Update;
import utils.Activation;

public interface Layer{
	public int nextSize();
	public int prevSize();
	public void init(int prevSize);
	public void init(int prevSize, double[][] weights, double[] bias);
	public double[] getBias();
	public Edge[] edges();
	public double[] forwardPropagate(double[] input);
	public Update backPropagate(int l, double[] prevResult, double[] nextResult, double[] error, double[] error2, double lambda, double weightSum, Optimizer optimizer, int size, int max, int max2);
	public void update();
	public Activation getActivation();
	public double getDropout();
	public int byteSize();
	public ByteBuffer toBytes();
}
