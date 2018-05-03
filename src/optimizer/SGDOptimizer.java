package optimizer;

import edge.Edge;
import utils.Activation;

public class SGDOptimizer implements Optimizer{
	private double learnRate;
	
	public SGDOptimizer(){
		this.learnRate = 0.01;
	}
	
	public SGDOptimizer(double learnRate){
		this.learnRate = learnRate;
	}
	
	@Override
	public double optimizeWeight(double grad, int l, Edge e, int size, int max, int nextSize){
		return -learnRate * grad;
	}
	
	@Override
	public double optimizeBias(double grad, int l, int i, int size, int max, int nextSize){
		return -learnRate * grad;
	}
}
