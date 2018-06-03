package javamachinelearning.layers.feedforward;

import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

public class ActivationLayer implements FeedForwardLayer{
	private int[] shape;
	private Activation activation;
	
	public ActivationLayer(Activation activation){
		this.activation = activation;
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
		return activation.activate(input);
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error){
		return error.mul(activation.derivative(output));
	}
	
	@Override
	public String toString(){
		return "Activation: " + activation.toString();
	}
}
