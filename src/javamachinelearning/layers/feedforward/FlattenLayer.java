package javamachinelearning.layers.feedforward;

import java.util.Arrays;

import javamachinelearning.utils.Tensor;

public class FlattenLayer implements FeedForwardLayer{
	private int[] inputShape;
	private int outputSize;
	
	public FlattenLayer(){
		// nothing to do
	}
	
	@Override
	public int[] outputShape(){
		return new int[]{1, outputSize};
	}
	
	@Override
	public int[] inputShape(){
		return inputShape;
	}
	
	@Override
	public void init(int[] inputShape){
		this.inputShape = inputShape;
		outputSize = 1;
		for(int i = 0; i < inputShape.length; i++){
			outputSize *= inputShape[i];
		}
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		return input.flatten();
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error){
		return error.reshape(inputShape);
	}
	
	@Override
	public String toString(){
		return "Flatten\tInput Shape: " + Arrays.toString(inputShape()) + "\tOutput Shape: " + Arrays.toString(outputShape());
	}
}
