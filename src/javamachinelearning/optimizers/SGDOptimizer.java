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
	public void init(int[][] weightShapes, int[][] biasShapes){
		// nothing to do
	}
	
	@Override
	public void update(){
		// nothing to do
	}
	
	@Override
	public Tensor optimizeWeight(Tensor grads, int l){
		return grads.mul(learnRate);
	}
	
	@Override
	public Tensor optimizeBias(Tensor grads, int l){
		return grads.mul(learnRate);
	}
}
