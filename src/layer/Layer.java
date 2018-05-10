package layer;

import java.nio.ByteBuffer;

import edge.Edge;
import optimizer.Optimizer;
import optimizer.Update;
import utils.Activation;
import utils.Tensor;

public interface Layer{
	public int nextSize();
	public int prevSize();
	public void init(int prevSize);
	public void init(int prevSize, double[][] weights, double[] bias);
	public Tensor bias();
	public Tensor weights();
	public Tensor forwardPropagate(Tensor input);
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, Optimizer optimizer, int l);
	public void update();
	public Activation getActivation();
	public double dropout();
	public int byteSize();
	public ByteBuffer bytes();
}
