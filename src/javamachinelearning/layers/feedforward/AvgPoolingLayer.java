package javamachinelearning.layers.feedforward;

import java.util.Arrays;

import javamachinelearning.utils.Tensor;

public class AvgPoolingLayer implements FeedForwardLayer{
	private int[] inputShape;
	private int[] outputShape;
	private int winWidth, winHeight;
	private int strideX, strideY;
	
	public AvgPoolingLayer(int winWidth, int winHeight, int strideX, int strideY){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
	}
	
	public AvgPoolingLayer(int winSize, int stride){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = stride;
		this.strideY = stride;
	}
	
	public AvgPoolingLayer(int winSize){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = 1;
		this.strideY = 1;
	}
	
	@Override
	public int[] outputShape(){
		return outputShape;
	}
	
	@Override
	public int[] inputShape(){
		return inputShape;
	}
	
	@Override
	public void init(int[] inputShape){
		this.inputShape = inputShape;
		
		int temp = inputShape[0] - winWidth;
		if(temp % strideX != 0)
			throw new IllegalArgumentException("Bad sizes for average pooling!");
		int w = temp / strideX + 1;
		
		temp = inputShape[1] - winHeight;
		if(temp % strideY != 0)
			throw new IllegalArgumentException("Bad sizes for average pooling!");
		int h = temp / strideY + 1;
		
		outputShape = new int[]{w, h, inputShape[2]};
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		double[] res = new double[outputShape[0] * outputShape[1] * outputShape[2]];
		int[] shape = input.shape();
		int idx = 0;
		// slide through and computes the average for each location
		// the output should have the same depth as the input
		for(int i = 0; i < outputShape[0] * strideX; i += strideX){
			for(int j = 0; j < outputShape[1] * strideY; j += strideY){
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
		
		return new Tensor(outputShape, res);
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error){
		double[] res = new double[inputShape[0] * inputShape[1] * inputShape[2]];
		int outIdx = 0;
		
		for(int i = 0; i < outputShape[0] * strideX; i += strideX){
			for(int j = 0; j < outputShape[1] * strideY; j += strideY){
				for(int k = 0; k < inputShape[2]; k++){ // for each depth slice
					for(int rx = 0; rx < winWidth; rx++){ // relative x position
						for(int ry = 0; ry < winHeight; ry++){ // relative y position
							// absolute positions
							int x = i + rx;
							int y = j + ry;
							int inIdx = x * inputShape[1] * inputShape[2] + y * inputShape[2] + k;
							
							res[inIdx] += error.flatGet(outIdx) / (winWidth * winHeight);
						}
					}
					
					outIdx++;
				}
			}
		}
		
		return new Tensor(inputShape, res);
	}
	
	@Override
	public String toString(){
		return "Average Pooling\tInput Shape: " + Arrays.toString(inputShape()) + "\tOutput Shape: " + Arrays.toString(outputShape());
	}
}
