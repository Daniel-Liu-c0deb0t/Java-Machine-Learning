package javamachinelearning.layers.feedforward;

import java.util.Random;

import javamachinelearning.utils.Tensor;

public class DropoutLayer implements FeedForwardLayer{
	private double dropout;
	private int[] shape;
	private int size;
	private Tensor mask;
	
	public DropoutLayer(){
		this.dropout = 0.5;
	}
	
	public DropoutLayer(double dropout){
		this.dropout = dropout;
	}
	
	@Override
	public int[] nextShape(){
		return shape;
	}
	
	@Override
	public int[] prevShape(){
		return shape;
	}
	
	@Override
	public void init(int[] prevSize){
		shape = prevSize;
		size = 1;
		for(int i = 0; i < shape.length; i++){
			size *= shape[i];
		}
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		if(training){
			double[] arr = new double[size];
			Random r = new Random();
			for(int i = 0; i < size; i++){
				// if not dropout, then scale the inputs
				arr[i] = r.nextDouble() < dropout ? 0.0 : (1.0 / (1.0 - dropout));
			}
			mask = new Tensor(shape, arr);
			
			return input.mul(mask);
		}else{
			// do not need to scale inputs
			return input;
		}
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error){
		// scale the gradients during backpropagation
		return error.mul(mask);
	}
}
