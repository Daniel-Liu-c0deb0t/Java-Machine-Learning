package layer;

import java.nio.ByteBuffer;

import optimizer.Optimizer;
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
	private double dropout;
	
	public FCLayer(int nextSize){
		this.nextSize = nextSize;
		this.activation = Activation.linear;
		this.dropout = 0.0;
	}
	
	public FCLayer(int nextSize, Activation activation){
		this.nextSize = nextSize;
		this.activation = activation;
		this.dropout = 0.0;
	}
	
	public FCLayer(int nextSize, Activation activation, double dropout){
		this.nextSize = nextSize;
		this.activation = activation;
		this.dropout = dropout;
	}
	
	@Override
	public int nextSize(){
		return nextSize;
	}
	
	@Override
	public int prevSize(){
		return prevSize;
	}

	@Override
	public void init(int prevSize){
		this.prevSize = prevSize;
		this.weights = new Tensor(new int[]{prevSize, nextSize}, true);
		this.deltaWeights = new Tensor(new int[]{prevSize, nextSize}, false);
		this.bias = new Tensor(new int[]{nextSize}, false);
		this.deltaBias = new Tensor(new int[]{nextSize}, false);
	}
	
	@Override
	public void init(int prevSize, double[][] weights, double[] bias){
		this.prevSize = prevSize;
		this.weights = new Tensor(weights);
		this.deltaWeights = new Tensor(new int[]{prevSize, nextSize}, false);
		this.bias = new Tensor(bias);
		this.deltaBias = new Tensor(new int[]{nextSize}, false);
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
	public Tensor forwardPropagate(Tensor input){
		return activation.activate(weights.dot(input).add(bias));
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error, double regLambda, int weightCount, Optimizer optimizer, int l){
		// error wrt layer output derivative
		Tensor grads = error.mul(activation.derivative(nextRes));
		
		// new error should be affected by weights
		Tensor nextError = weights.T().dot(grads);
		
		// error wrt weight derivative
		deltaWeights = deltaWeights.add(optimizer.optimizeWeight(prevRes.mulEach(grads), l).sub(weights.mul(regLambda / weightCount)));
		
		// error wrt bias derivative
		// not multiplied by prev outputs!
		deltaBias = deltaBias.add(optimizer.optimizeBias(grads, l));
		
		changeCount++;
		
		return nextError;
	}
	
	@Override
	public void update(){
		// handles postponed updates, by average updating values
		weights = weights.add(deltaWeights.div(Math.max(changeCount, 1)));
		deltaWeights = new Tensor(deltaWeights.shape(), 0);
		bias = bias.add(deltaBias.div(Math.max(changeCount, 1)));
		deltaBias = new Tensor(deltaBias.shape(), 0);
		changeCount = 0;
	}
	
	@Override
	public Activation getActivation(){
		return activation;
	}
	
	@Override
	public double dropout(){
		return dropout;
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
