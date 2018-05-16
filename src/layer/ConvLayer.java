package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import regularize.Regularizer;
import utils.Activation;
import utils.Tensor;

public class ConvLayer implements Layer{
	private Tensor weights;
	private Tensor deltaWeights;
	private Tensor bias;
	private Tensor deltaBias;
	
	private int[] prevSize;
	private int[] nextSize;
	private Activation activation;
	private int winWidth, winHeight;
	private int strideX, strideY;
	private int paddingX, paddingY;
	private int filterCount;
	private int changeCount;
	private boolean alreadyInit = false;
	
	public ConvLayer(int winWidth, int winHeight, int strideX, int strideY, int filterCount, int paddingX, int paddingY, Activation activation){
		this.winWidth = winWidth;
		this.winHeight = winHeight;
		this.strideX = strideX;
		this.strideY = strideY;
		this.filterCount = filterCount;
		this.activation = activation;
	}
	
	public ConvLayer(int winSize, int stride, int filterCount, int padding, Activation activation){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = stride;
		this.strideY = stride;
		this.filterCount = filterCount;
		this.activation = activation;
	}
	
	public ConvLayer(int winSize, int stride, int filterCount, int padding){
		this.winWidth = winSize;
		this.winHeight = winSize;
		this.strideX = stride;
		this.strideY = stride;
		this.filterCount = filterCount;
		this.activation = Activation.linear;
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
		
		int temp = prevSize[0] - winWidth + paddingX * 2;
		if(temp % strideX != 0)
			throw new IllegalArgumentException("Bad sizes for convolution!");
		int w = temp / strideX + 1;
		
		temp = prevSize[1] - winHeight + paddingY * 2;
		if(temp % strideY != 0)
			throw new IllegalArgumentException("Bad sizes for convolution!");
		int h = temp / strideY + 1;
		
		nextSize = new int[]{w, h, filterCount};
		
		if(!alreadyInit){
			weights = new Tensor(new int[]{winWidth, winHeight, prevSize[2], filterCount}, true);
			bias = new Tensor(new int[]{1, 1, filterCount}, false);
		}
		deltaWeights = new Tensor(new int[]{winWidth, winHeight, prevSize[2], filterCount}, false);
		deltaBias = new Tensor(new int[]{1, 1, filterCount}, false);
	}
	
	public ConvLayer withParams(Tensor w, Tensor b){
		weights = w;
		bias = b;
		alreadyInit = true;
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
	public Tensor forwardPropagate(Tensor input, boolean training){
		double[] res = new double[nextSize[0] * nextSize[1] * filterCount];
		int[] inMult = input.mult(); // the mult for prevSize because input shape equals prevSize
		int[] wMult = weights.mult();
		int idx = 0;
		
		for(int i = 0; i < nextSize[0] * strideX; i += strideX){
			for(int j = 0; j < nextSize[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < prevSize[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= prevSize[0] || y < 0 || y >= prevSize[1])
									continue;
								
								// multiply by weight and accumulate by addition
								res[idx] += input.flatGet(x * inMult[0] + y * inMult[1] + depth) *
										weights.flatGet(rx * wMult[0] + ry * wMult[1] + depth * wMult[2] + filter);
							}
						}
					}
					
					// add bias
					res[idx] += bias.flatGet(filter);
					
					idx++;
				}
			}
		}
		
		return activation.activate(new Tensor(nextSize, res));
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Optimizer optimizer, Regularizer regularizer, int l){
		Tensor grads = error.mul(activation.derivative(nextRes));
		
		// calculating delta weights and delta biases
		double[] deltaW = new double[weights.size()];
		double[] deltaB = new double[bias.size()];
		int[] inMult = prevRes.mult();
		int[] wMult = weights.mult();
		int gradIdx = 0;
		
		for(int i = 0; i < nextSize[0] * strideX; i += strideX){
			for(int j = 0; j < nextSize[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < prevSize[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= prevSize[0] || y < 0 || y >= prevSize[1])
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
					deltaB[filter] += grads.flatGet(gradIdx);
					
					gradIdx++;
				}
			}
		}
		
		deltaWeights = deltaWeights.sub(optimizer.optimizeWeight(new Tensor(weights.shape(), deltaW), l));
		if(regularizer != null)
			deltaWeights = deltaWeights.sub(regularizer.derivative(weights));
		
		deltaBias = deltaBias.sub(optimizer.optimizeBias(new Tensor(bias.shape(), deltaB), l));
		
		// calculate the next error gradients
		double[] nextError = new double[prevRes.size()];
		gradIdx = 0;
		
		for(int i = 0; i < nextSize[0] * strideX; i += strideX){
			for(int j = 0; j < nextSize[1] * strideY; j += strideY){
				for(int filter = 0; filter < filterCount; filter++){
					// relative to each filter
					for(int rx = 0; rx < winWidth; rx++){
						for(int ry = 0; ry < winHeight; ry++){
							for(int depth = 0; depth < prevSize[2]; depth++){
								// absolute positions
								int x = i - paddingX + rx;
								int y = j - paddingY + ry;
								
								// handle zero padding
								if(x < 0 || x >= prevSize[0] || y < 0 || y >= prevSize[1])
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
		
		return new Tensor(prevSize, nextError);
	}
	
	@Override
	public void update(){
		weights = weights.add(deltaWeights.div(Math.max(changeCount, 1)));
		deltaWeights = new Tensor(deltaWeights.shape(), false);
		bias = bias.add(deltaBias.div(Math.max(changeCount, 1)));
		deltaBias = new Tensor(deltaBias.shape(), false);
		changeCount = 0;
	}
	
	@Override
	public Activation getActivation(){
		return activation;
	}
	
	@Override
	public int byteSize(){
		return Double.BYTES * weights.size() + Double.BYTES * bias.size();
	}
	
	@Override
	public ByteBuffer bytes(){
		ByteBuffer bb = ByteBuffer.allocate(byteSize());
		for(int i = 0; i < weights.size(); i++){
			bb.putDouble(weights.flatGet(i));
		}
		for(int i = 0; i < bias.size(); i++){
			bb.putDouble(bias.flatGet(i));
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
		
		double[] b = new double[bias.size()];
		for(int i = 0; i < b.length; i++){
			b[i] = bb.getDouble();
		}
		bias = new Tensor(bias.shape(), b);
	}
}
