package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class SGDOptimizer implements Optimizer{
	private double learnRate;
	
	public SGDOptimizer(){
		this.learnRate = 0.01;
	}
	
	public SGDOptimizer(double learnRate){
		this.learnRate = learnRate;
	}
	
	@Override
	public int extraParams(){
		return 0;
	}
	
	@Override
	public void update(){
		// nothing to do
	}
	
	@Override
	public Tensor optimize(Tensor grads, Tensor[] params){
		return grads.mul(learnRate);
	}
}
