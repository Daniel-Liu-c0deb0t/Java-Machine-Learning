package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import utils.Activation;
import utils.Tensor;

public class FlattenLayer implements Layer{
	private int[] prevSize;
	private int nextSize;
	
	public FlattenLayer(){
		// nothing to do
	}
	
	@Override
	public int[] nextSize(){
		return new int[]{nextSize};
	}
	
	@Override
	public int[] prevSize(){
		return prevSize;
	}
	
	@Override
	public void init(int[] prevSize){
		this.prevSize = prevSize;
		nextSize = 1;
		for(int i = 0; i < prevSize.length; i++){
			nextSize *= prevSize[i];
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
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, Optimizer optimizer, int l){
		return error.reshape(prevSize);
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
