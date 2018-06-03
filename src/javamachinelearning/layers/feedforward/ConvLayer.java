package javamachinelearning.layers.feedforward;

import java.nio.ByteBuffer;
import java.util.Arrays;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Tensor;

public class ConvLayer implements FeedForwardParamsLayer{
	private Tensor weights;
	private Tensor gradWeights;
	private Tensor[] weightExtraParams;
	
	private Tensor bias;
	private Tensor gradBias;
	private Tensor[] biasExtraParams;
	
	private int[] inputShape;
	private int[] outputShape;
	private int winWidth, winHeight;
	private int strideX, strideY;
	private int paddingX, paddingY;
	private int filterCount;
	private int changeCount;
	private boolean alreadyInit = false;
	private boolean useBias = true;
	
	public ConvLayer(int winWidth, int winHeight, int strideX, int strideY, int filterCount, int paddingX, int paddingY){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
		this.filterCount = filterCount;
		this.paddingX = paddingX;
		this.paddingY = paddingY;
	}
	
	public ConvLayer(int winSize, int stride, int filterCount, int padding){
		this(winSize, winSize, stride, stride, filterCount, padding, padding);
	}
	
	public ConvLayer(int winSize, int filterCount, int padding){
		this(winSize, 1, filterCount, padding);
	}
	
	public ConvLayer(int winWidth, int winHeight, int strideX, int strideY, int filterCount, PaddingType type){
		if(type == PaddingType.VALID){
			this.winWidth = winWidth;
			this.winHeight = winHeight;
			this.strideX = strideX;
			this.strideY = strideY;
			this.filterCount = filterCount;
			this.paddingX = 0;
			this.paddingY = 0;
		}else{
			this.winWidth = winWidth;
			this.winHeight = winHeight;
			this.strideX = strideX;
			this.strideY = strideY;
			this.filterCount = filterCount;
			if((winWidth - 1) % 2 != 0)
				throw new IllegalArgumentException("Bad sizes for convolution!");
			this.paddingX = (winWidth - 1) / 2;
			if((winHeight - 1) % 2 != 0)
				throw new IllegalArgumentException("Bad sizes for convolution!");
			this.paddingY = (winHeight - 1) / 2;
		}
	}
	
	public ConvLayer(int winSize, int stride, int filterCount, PaddingType type){
		this(winSize, winSize, stride, stride, filterCount, type);
	}
	
	public ConvLayer(int winSize, int filterCount, PaddingType type){
		this(winSize, 1, filterCount, type);
	}
	
	public ConvLayer(int winSize, int filterCount){
		this(winSize, filterCount, PaddingType.VALID);
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
		
		int temp = inputShape[0] - winWidth + paddingX * 2;
		if(temp % strideX != 0)
			throw new IllegalArgumentException("Bad sizes for convolution!");
		int w = temp / strideX + 1;
		
		temp = inputShape[1] - winHeight + paddingY * 2;
		if(temp % strideY != 0)
			throw new IllegalArgumentException("Bad sizes for convolution!");
		int h = temp / strideY + 1;
		
		outputShape = new int[]{w, h, filterCount};
		
		if(!alreadyInit){
			weights = new Tensor(new int[]{winWidth, winHeight, inputShape[2], filterCount}, true);
			if(useBias)
				bias = new Tensor(new int[]{1, 1, filterCount}, false);
		}
		gradWeights = new Tensor(new int[]{winWidth, winHeight, inputShape[2], filterCount}, false);
		if(useBias)
			gradBias = new Tensor(new int[]{1, 1, filterCount}, false);
	}
	
	@Override
	public FeedForwardParamsLayer withParams(Tensor w, Tensor b){
		weights = w;
		if(useBias)
			bias = b;
		alreadyInit = true;
		return this;
	}
	
	@Override
	public FeedForwardParamsLayer noBias(){
		useBias = false;
		return this;
	}
	
	@Override
	public Tensor bias(){
		return bias;
	}
	
	@Override
	public Tensor weights(){
		return weights;
	}
	
	@Override
	public void setBias(Tensor b){
		if(useBias)
			bias = b;
	}
	
	@Override
	public void setWeights(Tensor w){
		weights = w;
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		double[] res = new double[outputShape[0] * outputShape[1] * filterCount];
		int[] inMult = input.mult(); // equals the mult for inputShape because input shape equals inputShape
		int[] wMult = weights.mult();
		int idx = 0;
		
		for(int i = 0; i < outputShape[0] * strideX; i += strideX){
			for(int j = 0; j < outputShape[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < inputShape[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= inputShape[0] || y < 0 || y >= inputShape[1])
									continue;
								
								// multiply by weight and accumulate by addition
								res[idx] += input.flatGet(x * inMult[0] + y * inMult[1] + depth) *
										weights.flatGet(rx * wMult[0] + ry * wMult[1] + depth * wMult[2] + filter);
							}
						}
					}
					
					// add bias
					if(useBias)
						res[idx] += bias.flatGet(filter);
					
					idx++;
				}
			}
		}
		
		return new Tensor(outputShape, res);
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error){
		// calculate weight gradients and bias gradients
		double[] deltaW = new double[weights.size()];
		double[] deltaB = new double[bias.size()];
		int[] inMult = input.mult();
		int[] wMult = weights.mult();
		int gradIdx = 0;
		
		for(int i = 0; i < outputShape[0] * strideX; i += strideX){
			for(int j = 0; j < outputShape[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < inputShape[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= inputShape[0] || y < 0 || y >= inputShape[1])
									continue;
								
								int wIdx = rx * wMult[0] + ry * wMult[1] + depth * wMult[2] + filter;
								
								// multiply gradients by previous layer's output
								// accumulate gradients for each weight
								deltaW[wIdx] += error.flatGet(gradIdx) *
										input.flatGet(x * inMult[0] + y * inMult[1] + depth);
							}
						}
					}
					
					// accumulate gradients for the biases
					// one bias per filter!
					if(useBias)
						deltaB[filter] += error.flatGet(gradIdx);
					
					gradIdx++;
				}
			}
		}
		
		gradWeights = gradWeights.add(new Tensor(weights.shape(), deltaW));
		
		if(useBias)
			gradBias = gradBias.add(new Tensor(bias.shape(), deltaB));
		
		// calculate the gradients wrt input
		double[] gradInputs = new double[input.size()];
		gradIdx = 0;
		
		for(int i = 0; i < outputShape[0] * strideX; i += strideX){
			for(int j = 0; j < outputShape[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < inputShape[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= inputShape[0] || y < 0 || y >= inputShape[1])
									continue;
								
								int inIdx = x * inMult[0] + y * inMult[1] + depth;
								
								// multiply gradients by each weight
								// accumulate gradients for each input
								gradInputs[inIdx] += error.flatGet(gradIdx) *
										weights.flatGet(rx * wMult[0] + ry * wMult[1] + depth * wMult[2] + filter);
							}
						}
					}
					
					gradIdx++;
				}
			}
		}
		
		changeCount++;
		
		return new Tensor(inputShape, gradInputs);
	}
	
	@Override
	public void update(Optimizer optimizer, Regularizer regularizer){
		if(weightExtraParams == null){
			weightExtraParams = new Tensor[optimizer.extraParams()];
			for(int i = 0; i < weightExtraParams.length; i++){
				weightExtraParams[i] = new Tensor(weights.shape(), false);
			}
			
			if(useBias){
				biasExtraParams = new Tensor[optimizer.extraParams()];
				for(int i = 0; i < biasExtraParams.length; i++){
					biasExtraParams[i] = new Tensor(bias.shape(), false);
				}
			}
		}
		
		if(regularizer == null){
			weights = weights.sub(
					optimizer.optimize(
							gradWeights.div(Math.max(changeCount, 1)), weightExtraParams));
		}else{
			weights = weights.sub(
					optimizer.optimize(
							gradWeights.div(Math.max(changeCount, 1)).add(
									regularizer.derivative(weights)), weightExtraParams));
		}
		gradWeights = new Tensor(gradWeights.shape(), false);
		
		if(useBias){
			bias = bias.sub(
					optimizer.optimize(
							gradBias.div(Math.max(changeCount, 1)), biasExtraParams));
			gradBias = new Tensor(gradBias.shape(), false);
		}
		changeCount = 0;
	}
	
	@Override
	public int byteSize(){
		return Double.BYTES * weights.size() + (useBias ? Double.BYTES * bias.size() : 0);
	}
	
	@Override
	public ByteBuffer bytes(){
		ByteBuffer bb = ByteBuffer.allocate(byteSize());
		for(int i = 0; i < weights.size(); i++){
			bb.putDouble(weights.flatGet(i));
		}
		if(useBias){
			for(int i = 0; i < bias.size(); i++){
				bb.putDouble(bias.flatGet(i));
			}
		}
		bb.flip();
		return bb;
	}
	
	@Override
	public void readBytes(ByteBuffer bb){
		double[] w = new double[weights.size()];
		for(int i = 0; i < w.length; i++){
			w[i] = bb.getDouble();
		}
		weights = new Tensor(weights.shape(), w);
		
		if(useBias){
			double[] b = new double[bias.size()];
			for(int i = 0; i < b.length; i++){
				b[i] = bb.getDouble();
			}
			bias = new Tensor(bias.shape(), b);
		}
	}
	
	@Override
	public String toString(){
		return "Convolutional\tInput Shape: " + Arrays.toString(inputShape()) + "\tOutput Shape: " + Arrays.toString(outputShape());
	}
	
	public enum PaddingType{
		VALID, SAME;
	}
}
