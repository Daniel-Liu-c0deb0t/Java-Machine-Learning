package optimizer;

import utils.Tensor;

public class AdamOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	private double beta1;
	private double beta2;
	
	private Tensor[] mWeight;
	private Tensor[] vWeight;
	private Tensor[] mBias;
	private Tensor[] vBias;
	private double currBeta1; // these biases are changed while optimizing
	private double currBeta2;
	
	public AdamOptimizer(){
		this.learnRate = 0.01;
		this.beta1 = 0.9;
		this.beta2 = 0.999;
	}
	
	public AdamOptimizer(double learnRate){
		this.learnRate = learnRate;
		this.beta1 = 0.9;
		this.beta2 = 0.999;
	}
	
	public AdamOptimizer(double learnRate, double beta1, double beta2){
		this.learnRate = learnRate;
		this.beta1 = beta1;
		this.beta2 = beta2;
	}
	
	@Override
	public void init(int[][] weightShapes, int[][] biasShapes){
		mWeight = new Tensor[weightShapes.length];
		vWeight = new Tensor[weightShapes.length];
		mBias = new Tensor[weightShapes.length];
		vBias = new Tensor[weightShapes.length];
		
		for(int i = 0; i < weightShapes.length; i++){
			mWeight[i] = new Tensor(weightShapes[i], false);
			vWeight[i] = new Tensor(weightShapes[i], false);
			mBias[i] = new Tensor(biasShapes[i], false);
			vBias[i] = new Tensor(biasShapes[i], false);
		}
		
		currBeta1 = beta1;
		currBeta2 = beta2;
	}
	
	@Override
	public void update(){
		currBeta1 *= currBeta1;
		currBeta2 *= currBeta2;
	}
	
	@Override
	public Tensor optimizeWeight(Tensor grads, int l){
		mWeight[l] = mWeight[l].mul(beta1).add(grads.mul(1.0 - beta1));
		vWeight[l] = vWeight[l].mul(beta2).add(grads.mul(grads).mul(1.0 - beta2));
		return mWeight[l].div(1.0 - currBeta1).div(
				vWeight[l].div(1.0 - currBeta2).map(x -> Math.sqrt(x)).add(epsilon)).mul(-learnRate);
	}
	
	@Override
	public Tensor optimizeBias(Tensor grads, int l){
		mBias[l] = mBias[l].mul(beta1).add(grads.mul(1.0 - beta1));
		vBias[l] = vBias[l].mul(beta2).add(grads.mul(grads).mul(1.0 - beta2));
		return mBias[l].div(1.0 - currBeta1).div(
				vBias[l].div(1.0 - currBeta2).map(x -> Math.sqrt(x)).add(epsilon)).mul(-learnRate);
	}
}
