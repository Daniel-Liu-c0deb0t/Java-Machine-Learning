package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class RMSPropOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	private double mu;
	
	public RMSPropOptimizer(){
		this.mu = 0.9;
		this.learnRate = 0.1;
	}
	
	public RMSPropOptimizer(double learnRate){
		this.mu = 0.9;
		this.learnRate = learnRate;
	}
	
	@Override
	public int extraParams(){
		return 1;
	}
	
	@Override
	public void update(){
		// nothing to do
	}
	@Override
	public Tensor optimize(Tensor grads, Tensor[] params){
		// parameter: exponential average of squared gradients
		params[0] = params[0].mul(mu).add((grads.mul(grads)).mul(1.0-mu));
		return grads.mul(learnRate).div(params[0].map(x -> Math.sqrt(x)).add(epsilon));
	}
}
