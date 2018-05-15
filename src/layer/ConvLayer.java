package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
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
		return activation.activate(
				input.convolve(weights, bias, nextSize, winWidth, winHeight, strideX, strideY, paddingX, paddingY));
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, Optimizer optimizer, int l){
		Tensor grads = error.mul(activation.derivative(nextRes));
		deltaWeights = deltaWeights.sub(
				optimizer.optimizeWeight(
						prevRes.convolve(grads.reshape(nextSize[0], nextSize[1], 1, nextSize[2]), null,
								weights.shape(), nextSize[0], nextSize[1], strideX, strideY, paddingX, paddingY), l)
				.add(weights.mul(regLambda)));
		
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
