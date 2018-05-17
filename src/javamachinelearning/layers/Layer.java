package javamachinelearning.layers;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

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
