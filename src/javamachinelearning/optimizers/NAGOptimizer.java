package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class NAGOptimizer implements Optimizer{
	private double learnRate;
	private double mu; // friction to decay momentum

	public NAGOptimizer(){
		this.learnRate = 0.1;
		this.mu = 0.9;
	}

	public NAGOptimizer(double learnRate){
		this.learnRate = learnRate;
		this.mu = 0.9;
	}

	public NAGOptimizer(double learnRate, double mu){
		this.learnRate = learnRate;
		this.mu = mu;
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
		// parameter: velocity
		params[0] = params[0].mul(mu).sub(grads.mul(learnRate));
		return params[0].mul(-1.0); // is negated
	}
}
