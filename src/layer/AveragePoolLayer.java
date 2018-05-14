package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import utils.Activation;
import utils.Tensor;

public class AveragePoolLayer implements Layer{
	private int[] prevSize;
	private int[] nextSize;
	private int winWidth, winHeight;
	private int strideX, strideY;
	
	public AveragePoolLayer(int winWidth, int winHeight, int strideX, int strideY){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
	}
	
	public AveragePoolLayer(int winSize, int stride){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = stride;
		this.strideY = stride;
	}
	
	@Override
	public int[] nextSize(){
		return nextSize;
	}
	
	@Override
	public int[] prevSize(){
		return prevSize;
	}
	
	@Override
	public void init(int[] prevSize){
		this.prevSize = prevSize;
		
		int temp = prevSize[0] - winWidth + 1;
		int w = temp / strideX + (temp % strideX == 0 ? 0 : 1); //round up if not divisible
		
		temp = prevSize[1] - winHeight + 1;
		int h = temp / strideY + (temp % strideY == 0 ? 0 : 1);
		
		nextSize = new int[]{w, h};
	}
	
	@Override
	public void init(int[] prevSize, double[][] weights, double[] bias){
		// should not be used!
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
		double[] res = new double[nextSize[0] * nextSize[1]];
		int[] shape = input.shape();
		int idx = 0;
		// slide through and computes the averages
		// the output should have the same depth as the input
		for(int i = 0; i < shape[0]; i += strideX){
			for(int j = 0; j < shape[1]; j += strideY){
				for(int k = 0; k < shape[2]; k++){ // for each depth slice
					double sum = 0;
					int count = 0;
					
					for(int rx = 0; rx < Math.min(winWidth, shape[0] - i); rx++){ // relative x position
						for(int ry = 0; ry < Math.min(winHeight, shape[1] - j); ry++){ // relative y position
							// absolute positions
							int x = i + rx;
							int y = j + ry;
							
							sum += input.flatGet(x * shape[1] * shape[2] + y * shape[2] + k);
							count++;
						}
					}
					
					res[idx] = sum / count;
				}
			}
		}
		
		return new Tensor(nextSize, res);
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, int weightCount, Optimizer optimizer, int l){
		
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
