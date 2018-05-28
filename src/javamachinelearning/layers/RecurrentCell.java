package javamachinelearning.layers;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

public interface RecurrentCell{
	public void noBias();
	
	public int[] nextSize();
	public int[] prevSize();
	public Tensor forwardPropagate(Tensor input, Tensor prevState, boolean training);
	// backpropagation should return two tensors for the input and the previous state
	public Tensor[] backPropagate(Tensor input, Tensor prevState, Tensor error);
	public void update(Optimizer optimizer, Regularizer regularizer);
	public int byteSize();
	public ByteBuffer bytes();
	public void readBytes(ByteBuffer bb);
}
