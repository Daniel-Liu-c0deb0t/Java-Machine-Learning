package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import regularize.Regularizer;
import utils.Activation;
import utils.Tensor;

public class AvgPoolLayer implements Layer{
	private int[] prevSize;
	private int[] nextSize;
	private int winWidth, winHeight;
	private int strideX, strideY;
	
	public AvgPoolLayer(int winWidth, int winHeight, int strideX, int strideY){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
	}
	
	public AvgPoolLayer(int winSize, int stride){
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
		
		int temp = prevSize[0] - winWidth;
		if(temp % strideX != 0)
			throw new IllegalArgumentException("Bad sizes for average pooling!");
		int w = temp / strideX + 1;
		
		temp = prevSize[1] - winHeight;
		if(temp % strideY != 0)
			throw new IllegalArgumentException("Bad sizes for average pooling!");
		int h = temp / strideY + 1;
		
		nextSize = new int[]{w, h, prevSize[2]};
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
		double[] res = new double[nextSize[0] * nextSize[1] * nextSize[2]];
		int[] shape = input.shape();
		int idx = 0;
		// slide through and computes the average for each location
		// the output should have the same depth as the input
		for(int i = 0; i < nextSize[0] * strideX; i += strideX){
			for(int j = 0; j < nextSize[1] * strideY; j += strideY){
				for(int k = 0; k < shape[2]; k++){ // for each depth slice
					double sum = 0;
					
					for(int rx = 0; rx < winWidth; rx++){ // relative x position
						for(int ry = 0; ry < winHeight; ry++){ // relative y position
							// absolute positions
							int x = i + rx;
							int y = j + ry;
							
							sum += input.flatGet(x * shape[1] * shape[2] + y * shape[2] + k);
						}
					}
					
					// average of all values
					res[idx] = sum / (winWidth * winHeight);
					idx++;
				}
			}
		}
		
		return new Tensor(nextSize, res);
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Optimizer optimizer, Regularizer regularizer, int l){
		double[] res = new double[prevSize[0] * prevSize[1] * prevSize[2]];
		int outIdx = 0;
		
		for(int i = 0; i < nextSize[0] * strideX; i += strideX){
			for(int j = 0; j < nextSize[1] * strideY; j += strideY){
				for(int k = 0; k < prevSize[2]; k++){ // for each depth slice
					for(int rx = 0; rx < winWidth; rx++){ // relative x position
						for(int ry = 0; ry < winHeight; ry++){ // relative y position
							// absolute positions
							int x = i + rx;
							int y = j + ry;
							int inIdx = x * prevSize[1] * prevSize[2] + y * prevSize[2] + k;
							
							res[inIdx] += error.flatGet(outIdx) / (winWidth * winHeight);
						}
					}
					
					outIdx++;
				}
			}
		}
		
		return new Tensor(prevSize, res);
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
