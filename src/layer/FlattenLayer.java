package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import regularize.Regularizer;
import utils.Activation;
import utils.Tensor;

public class FlattenLayer implements Layer{
	private int[] prevShape;
	private int nextSize;
	
	public FlattenLayer(){
		// nothing to do
	}
	
	@Override
	public int[] nextShape(){
		return new int[]{nextSize};
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
	public Tensor bias(){
		return null;
	}

	@Override
	public Tensor weights(){
		return null;
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		return input.flatten();
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Optimizer optimizer, Regularizer regularizer, int l){
		return error.reshape(prevShape);
	}
	
	@Override
	public void update(){
		// nothing to do
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
		// nothing to do
	}
}
