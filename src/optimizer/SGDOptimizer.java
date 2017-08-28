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
	public double optimizeWeight(int l, Edge e, double[] prevResult, double[] nextResult, double[] error, double lambda, double weightSum, Activation activation, int size, int max, int nextSize){
		double g = (error[e.getNodeB()] + lambda * weightSum) * prevResult[e.getNodeA()] * activation.derivative(nextResult[e.getNodeB()]);
		return -learnRate * (g + lambda * e.getWeight());
	}
	
	@Override
	public double optimizeBias(int l, int i, double[] nextResult, double[] error, Activation activation, int size, int max){
		double g = error[i] * activation.derivative(nextResult[i]);
		return -learnRate * g;
	}
}
