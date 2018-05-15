package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import utils.Activation;
import utils.Tensor;

public interface Layer{
	public int[] nextSize();
	public int[] prevSize();
	public void init(int[] prevSize);
	public Tensor bias();
	public Tensor weights();
	public Tensor forwardPropagate(Tensor input, boolean training);
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, Optimizer optimizer, int l);
	public void update();
	public Activation getActivation();
	public int byteSize();
	public ByteBuffer bytes();
	public void readBytes(ByteBuffer bb);
}
