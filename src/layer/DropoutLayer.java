package layer;

import java.nio.ByteBuffer;
import java.util.Random;

import optimizer.Optimizer;
import utils.Activation;
import utils.Tensor;

public class DropoutLayer implements Layer{
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
	public int[] nextSize(){
		return shape;
	}
	
	@Override
	public int[] prevSize(){
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
	public Tensor bias(){
		return null;
	}
	
	@Override
	public Tensor weights(){
		return null;
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
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, Optimizer optimizer, int l){
		// scale the gradients during backprop
		return error.mul(mask);
	}
	
	@Override
	public void update(){
		// do nothing
	}
	
	@Override
	public Activation getActivation(){
		return null;
	}
	
	@Override
	public int byteSize(){
		return 0;
	}
	
	@Override
	public ByteBuffer bytes(){
		return null;
	}
	
	@Override
	public void readBytes(ByteBuffer bb){
		// do nothing
	}
}
