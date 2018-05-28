package javamachinelearning.layers.feedforward;

import javamachinelearning.layers.ParamsLayer;
import javamachinelearning.utils.Tensor;

public interface FeedForwardParamsLayer extends FeedForwardLayer, ParamsLayer{
	// withParams should be used when initializing a layer
	public FeedForwardParamsLayer withParams(Tensor w, Tensor b);
	
	public Tensor bias();
	public Tensor weights();
	public void setBias(Tensor b);
	public void setWeights(Tensor w);
}
