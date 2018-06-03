package javamachinelearning.layers.recurrent;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Tensor;

public interface RecurrentCell{
	public void noBias();
	
	public int[] outputShape();
	public int[] inputShape();
	public void init(int inputSize, int numTimeSteps);
	public Tensor forwardPropagate(int t, Tensor input, Tensor prevState, boolean training);
	// backpropagation should return two tensors for the input and the previous state gradients
	public Tensor[] backPropagate(int t, Tensor input, Tensor prevState, Tensor error);
	public void update(Optimizer optimizer, Regularizer regularizer, int changeCount);
	public int byteSize();
	public ByteBuffer bytes();
	public void readBytes(ByteBuffer bb);
}
