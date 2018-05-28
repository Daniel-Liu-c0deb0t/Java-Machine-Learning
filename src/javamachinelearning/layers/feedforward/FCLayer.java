package javamachinelearning.layers.feedforward;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

public class FCLayer implements FeedForwardParamsLayer{
	private Tensor weights;
	private Tensor gradWeights;
	private Tensor[] weightExtraParams; // extra optimization parameters for weights
	
	private Tensor bias;
	private Tensor gradBias;
	private Tensor[] biasExtraParams; // extra optimization parameters for biases
	
	private Activation activation;
	private int prevSize;
	private int nextSize;
	private int changeCount;
	private boolean alreadyInit = false;
	private boolean useBias = true;
	
	public FCLayer(int nextSize, Activation activation){
		this.nextSize = nextSize;
		this.activation = activation;
	}
	
	@Override
	public int[] nextShape(){
		return new int[]{nextSize};
	}
	
	@Override
	public int[] prevShape(){
		return new int[]{prevSize};
	}

	@Override
	public void init(int[] prevSize){
		this.prevSize = prevSize[0];
		if(!alreadyInit){
			this.weights = new Tensor(new int[]{prevSize[0], nextSize}, true);
			if(useBias)
				this.bias = new Tensor(new int[]{nextSize}, false);
		}
		this.gradWeights = new Tensor(new int[]{prevSize[0], nextSize}, false);
		if(useBias)
			this.gradBias = new Tensor(new int[]{nextSize}, false);
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
		Tensor x = weights.dot(input);
		if(useBias)
			x = x.add(bias);
		return activation.activate(x);
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error){
		// error wrt layer output derivative
		Tensor grads = error.mul(activation.derivative(nextRes));
		
		// error wrt weight derivative
		gradWeights = gradWeights.add(prevRes.mulEach(grads));
		
		// error wrt bias derivative
		// not multiplied by previous outputs!
		if(useBias)
			gradBias = gradBias.add(grads);
		
		// new error should be affected by weights
		Tensor gradInputs = weights.T().dot(grads);
		
		changeCount++;
		
		return gradInputs;
	}
	
	@Override
	public void update(Optimizer optimizer, Regularizer regularizer){
		// initialize extra parameters
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
		
		// handles postponed updates, by averaging accumulated gradients
		// add the regularization derivative if needed
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
		// 8 bytes for each double
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
}
