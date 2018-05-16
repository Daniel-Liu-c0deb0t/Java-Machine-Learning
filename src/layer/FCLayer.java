package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
import regularize.Regularizer;
import utils.Activation;
import utils.Tensor;

public class FCLayer implements Layer{
	private Tensor weights;
	private Tensor deltaWeights;
	private Tensor bias;
	private Tensor deltaBias;
	
	private Activation activation;
	private int prevSize;
	private int nextSize;
	private int changeCount;
	private boolean alreadyInit = false;
	
	public FCLayer(int nextSize){
		this.nextSize = nextSize;
		this.activation = Activation.linear;
	}
	
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
			this.bias = new Tensor(new int[]{nextSize}, false);
		}
		this.deltaWeights = new Tensor(new int[]{prevSize[0], nextSize}, false);
		this.deltaBias = new Tensor(new int[]{nextSize}, false);
	}
	
	public FCLayer withParams(Tensor w, Tensor b){
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
		return activation.activate(weights.dot(input).add(bias));
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, Optimizer optimizer, Regularizer regularizer, int l){
		// error wrt layer output derivative
		Tensor grads = error.mul(activation.derivative(nextRes));
		
		// error wrt weight derivative
		deltaWeights = deltaWeights.sub(optimizer.optimizeWeight(prevRes.mulEach(grads), l));
		if(regularizer != null) // also subtract the regularization derivative if necessary
			deltaWeights = deltaWeights.sub(regularizer.derivative(weights));
		
		// error wrt bias derivative
		// not multiplied by prev outputs!
		deltaBias = deltaBias.sub(optimizer.optimizeBias(grads, l));
		
		// new error should be affected by weights
		Tensor nextError = weights.T().dot(grads);
		
		changeCount++;
		
		return nextError;
	}
	
	@Override
	public void update(){
		// handles postponed updates, by average updating values
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
		// 8 bytes for each double
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
