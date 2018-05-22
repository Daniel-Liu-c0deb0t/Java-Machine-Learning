package javamachinelearning.layers;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

public interface ParamsLayer extends Layer{
	// withParams should be used when initializing a layer
	public ParamsLayer withParams(Tensor w, Tensor b);
	// if biases shouldn't be used
	public ParamsLayer noBias();
	
	public Tensor bias();
	public Tensor weights();
	public void setBias(Tensor b);
	public void setWeights(Tensor w);
	public void update(Optimizer optimizer, int l);
	public Activation getActivation();
	public int byteSize();
	public ByteBuffer bytes();
	public void readBytes(ByteBuffer bb);
}
