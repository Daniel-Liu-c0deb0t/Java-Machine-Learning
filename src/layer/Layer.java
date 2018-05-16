package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import regularize.Regularizer;
import utils.Activation;
import utils.Tensor;

public interface Layer{
	public int[] nextShape();
	public int[] prevShape();
	public void init(int[] prevSize);
	public Tensor bias();
	public Tensor weights();
	public Tensor forwardPropagate(Tensor input, boolean training);
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Optimizer optimizer, Regularizer regularizer, int l);
	public void update();
	public Activation getActivation();
	public int byteSize();
	public ByteBuffer bytes();
	public void readBytes(ByteBuffer bb);
}
