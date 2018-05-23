package javamachinelearning.layers;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

public class ConvLayer implements ParamsLayer{
	private Tensor weights;
	private Tensor deltaWeightGrads;
	private Tensor bias;
	private Tensor deltaBiasGrads;
	
	private int[] prevShape;
	private int[] nextShape;
	private Activation activation;
	private int winWidth, winHeight;
	private int strideX, strideY;
	private int paddingX, paddingY;
	private int filterCount;
	private int changeCount;
	private boolean alreadyInit = false;
	private boolean useBias = true;
	
	public ConvLayer(int winWidth, int winHeight, int strideX, int strideY, int filterCount, int paddingX, int paddingY, Activation activation){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
		this.filterCount = filterCount;
		this.paddingX = paddingX;
		this.paddingY = paddingY;
		this.activation = activation;
	}
	
	public ConvLayer(int winSize, int stride, int filterCount, int padding, Activation activation){
		this(winSize, winSize, stride, stride, filterCount, padding, padding, activation);
	}
	
	public ConvLayer(int winSize, int filterCount, int padding, Activation activation){
		this(winSize, 1, filterCount, padding, activation);
	}
	
	public ConvLayer(int winWidth, int winHeight, int strideX, int strideY, int filterCount, PaddingType type, Activation activation){
		if(type == PaddingType.VALID){
			this.winWidth = winWidth;
			this.winHeight = winHeight;
			this.strideX = strideX;
			this.strideY = strideY;
			this.filterCount = filterCount;
			this.paddingX = 0;
			this.paddingY = 0;
			this.activation = activation;
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
			this.activation = activation;
		}
	}
	
	public ConvLayer(int winSize, int stride, int filterCount, PaddingType type, Activation activation){
		this(winSize, winSize, stride, stride, filterCount, type, activation);
	}
	
	public ConvLayer(int winSize, int filterCount, PaddingType type, Activation activation){
		this(winSize, 1, filterCount, type, activation);
	}
	
	public ConvLayer(int winSize, int filterCount, Activation activation){
		this(winSize, filterCount, PaddingType.VALID, activation);
	}
	
	@Override
	public int[] nextShape(){
		return nextShape;
	}
	
	@Override
	public int[] prevShape(){
		return prevShape;
	}
	
	@Override
	public void init(int[] prevShape){
		this.prevShape = prevShape;
		
		int temp = prevShape[0] - winWidth + paddingX * 2;
		if(temp % strideX != 0)
			throw new IllegalArgumentException("Bad sizes for convolution!");
		int w = temp / strideX + 1;
		
		temp = prevShape[1] - winHeight + paddingY * 2;
		if(temp % strideY != 0)
			throw new IllegalArgumentException("Bad sizes for convolution!");
		int h = temp / strideY + 1;
		
		nextShape = new int[]{w, h, filterCount};
		
		if(!alreadyInit){
			weights = new Tensor(new int[]{winWidth, winHeight, prevShape[2], filterCount}, true);
			if(useBias)
				bias = new Tensor(new int[]{1, 1, filterCount}, false);
		}
		deltaWeightGrads = new Tensor(new int[]{winWidth, winHeight, prevShape[2], filterCount}, false);
		if(useBias)
			deltaBiasGrads = new Tensor(new int[]{1, 1, filterCount}, false);
	}
	
	@Override
	public ParamsLayer withParams(Tensor w, Tensor b){
		weights = w;
		if(useBias)
			bias = b;
		alreadyInit = true;
		return this;
	}
	
	@Override
	public ParamsLayer noBias(){
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
		double[] res = new double[nextShape[0] * nextShape[1] * filterCount];
		int[] inMult = input.mult(); // the mult for prevShape because input shape equals prevShape
		int[] wMult = weights.mult();
		int idx = 0;
		
		for(int i = 0; i < nextShape[0] * strideX; i += strideX){
			for(int j = 0; j < nextShape[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < prevShape[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= prevShape[0] || y < 0 || y >= prevShape[1])
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
		
		return activation.activate(new Tensor(nextShape, res));
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Regularizer regularizer){
		Tensor grads = error.mul(activation.derivative(nextRes));
		
		// calculate weight gradients and bias gradients
		double[] deltaW = new double[weights.size()];
		double[] deltaB = new double[bias.size()];
		int[] inMult = prevRes.mult();
		int[] wMult = weights.mult();
		int gradIdx = 0;
		
		for(int i = 0; i < nextShape[0] * strideX; i += strideX){
			for(int j = 0; j < nextShape[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < prevShape[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= prevShape[0] || y < 0 || y >= prevShape[1])
									continue;
								
								int wIdx = rx * wMult[0] + ry * wMult[1] + depth * wMult[2] + filter;
								
								// multiply gradients by previous layer's output
								// accumulate gradients for each weight
								deltaW[wIdx] += grads.flatGet(gradIdx) *
										prevRes.flatGet(x * inMult[0] + y * inMult[1] + depth);
							}
						}
					}
					
					// accumulate gradients for the biases
					// one bias per filter!
					if(useBias)
						deltaB[filter] += grads.flatGet(gradIdx);
					
					gradIdx++;
				}
			}
		}
		
		if(regularizer == null){
			deltaWeightGrads = deltaWeightGrads.add(new Tensor(weights.shape(), deltaW));
		}else{ // also add the regularization derivative if necessary
			deltaWeightGrads = deltaWeightGrads.add(
					new Tensor(weights.shape(), deltaW).add(regularizer.derivative(weights)));
		}
		
		if(useBias)
			deltaBiasGrads = deltaBiasGrads.add(new Tensor(bias.shape(), deltaB));
		
		// calculate the next error gradients
		double[] nextError = new double[prevRes.size()];
		gradIdx = 0;
		
		for(int i = 0; i < nextShape[0] * strideX; i += strideX){
			for(int j = 0; j < nextShape[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < prevShape[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= prevShape[0] || y < 0 || y >= prevShape[1])
									continue;
								
								int inIdx = x * inMult[0] + y * inMult[1] + depth;
								
								// multiply gradients by each weight
								// accumulate gradients for each input
								nextError[inIdx] += grads.flatGet(gradIdx) *
										weights.flatGet(rx * wMult[0] + ry * wMult[1] + depth * wMult[2] + filter);
							}
						}
					}
					
					gradIdx++;
				}
			}
		}
		
		changeCount++;
		
		return new Tensor(prevShape, nextError);
	}
	
	@Override
	public void update(Optimizer optimizer, int l){
		weights = weights.sub(
				optimizer.optimizeWeight(deltaWeightGrads.div(Math.max(changeCount, 1)), l));
		deltaWeightGrads = new Tensor(deltaWeightGrads.shape(), false);
		if(useBias){
			bias = bias.sub(
					optimizer.optimizeBias(deltaBiasGrads.div(Math.max(changeCount, 1)), l));
			deltaBiasGrads = new Tensor(deltaBiasGrads.shape(), false);
		}
		changeCount = 0;
	}
	
	@Override
	public Activation getActivation(){
		return activation;
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
	
	public enum PaddingType{
		VALID, SAME;
	}
}
