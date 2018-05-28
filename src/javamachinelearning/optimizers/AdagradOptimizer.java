package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class AdagradOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	
	public AdagradOptimizer(){
		this.learnRate = 0.1;
	}
	
	public AdagradOptimizer(double learnRate){
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
		// parameter: sum of squared gradients
		params[0] = params[0].add(grads.mul(grads));
		return grads.mul(learnRate).div(params[0].map(x -> Math.sqrt(x)).add(epsilon));
	}
}
