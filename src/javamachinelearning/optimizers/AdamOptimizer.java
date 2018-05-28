package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class AdamOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	private double beta1;
	private double beta2;
	
	private double currBeta1; // these biases are changed while optimizing
	private double currBeta2;
	
	public AdamOptimizer(){
		this.learnRate = 0.001;
		this.beta1 = 0.9;
		this.beta2 = 0.999;
		
		currBeta1 = this.beta1;
		currBeta2 = this.beta2;
	}
	
	public AdamOptimizer(double learnRate){
		this.learnRate = learnRate;
		this.beta1 = 0.9;
		this.beta2 = 0.999;
		
		currBeta1 = this.beta1;
		currBeta2 = this.beta2;
	}
	
	public AdamOptimizer(double learnRate, double beta1, double beta2){
		this.learnRate = learnRate;
		this.beta1 = beta1;
		this.beta2 = beta2;
		
		currBeta1 = this.beta1;
		currBeta2 = this.beta2;
	}
	
	@Override
	public int extraParams(){
		return 2;
	}
	
	@Override
	public void update(){
		currBeta1 *= beta1;
		currBeta2 *= beta2;
	}
	
	@Override
	public Tensor optimize(Tensor grads, Tensor[] params){
		// parameter 1: momentum
		// parameter 2: velocity
		params[0] = params[0].mul(beta1).add(grads.mul(1.0 - beta1));
		params[1] = params[1].mul(beta2).add(grads.mul(grads).mul(1.0 - beta2));
		return params[0].div(1.0 - currBeta1).div(
				params[1].div(1.0 - currBeta2).map(x -> Math.sqrt(x)).add(epsilon)).mul(learnRate);
	}
}
