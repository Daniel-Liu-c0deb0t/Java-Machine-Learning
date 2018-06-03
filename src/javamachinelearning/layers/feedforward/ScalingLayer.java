package javamachinelearning.layers.feedforward;

import javamachinelearning.utils.Tensor;

public class ScalingLayer implements FeedForwardLayer{
	private int[] shape;
	private double scale;
	private boolean useTraining;
	
	public ScalingLayer(double scale, boolean useTraining){
		this.scale = scale;
		this.useTraining = useTraining;
	}
	
	@Override
	public int[] outputShape(){
		return shape;
	}
	
	@Override
	public int[] inputShape(){
		return shape;
	}
	
	@Override
	public void init(int[] inputShape){
		shape = inputShape;
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		if(!training || useTraining)
			return input.mul(scale);
		else
			return input;
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error){
		if(useTraining)
			return error.mul(scale);
		else
			return error;
	}
	
	@Override
	public String toString(){
		return "Scaling Factor: " + scale;
	}
}
