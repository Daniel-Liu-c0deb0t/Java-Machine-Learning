package javamachinelearning.layers;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;

public interface ParamsLayer extends Layer{
	// if biases shouldn't be used
	public ParamsLayer noBias();
	
	public void update(Optimizer optimizer, Regularizer regularizer);
	public int byteSize();
	public ByteBuffer bytes();
	public void readBytes(ByteBuffer bb);
}
