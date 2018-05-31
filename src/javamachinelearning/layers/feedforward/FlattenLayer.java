package javamachinelearning.layers.feedforward;

import javamachinelearning.utils.Tensor;

public class FlattenLayer implements FeedForwardLayer{
	private int[] prevShape;
	private int nextSize;
	
	public FlattenLayer(){
		// nothing to do
	}
	
	@Override
	public int[] nextShape(){
		return new int[]{1, nextSize};
	}
	
	@Override
	public int[] prevShape(){
		return prevShape;
	}
	
	@Override
	public void init(int[] prevShape){
		this.prevShape = prevShape;
		nextSize = 1;
		for(int i = 0; i < prevShape.length; i++){
			nextSize *= prevShape[i];
		}
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		return input.flatten();
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error){
		return error.reshape(prevShape);
	}
}
