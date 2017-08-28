package optimizer;

import edge.Edge;
import utils.Activation;

public class AdagradOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	
	private double[][] h = null;
	private double[][] hb = null;
	
	public AdagradOptimizer(){
		this.learnRate = 0.01;
	}
	
	public AdagradOptimizer(double learnRate){
		this.learnRate = learnRate;
	}
	
	@Override
	public double optimizeWeight(int l, Edge e, double[] prevResult, double[] nextResult, double[] error, double lambda, double weightSum, Activation activation, int size, int max, int nextSize){
		if(h == null){
			h = new double[size][max];
		}
		double g = (error[e.getNodeB()] + lambda * weightSum) * prevResult[e.getNodeA()] * activation.derivative(nextResult[e.getNodeB()]);
		h[l][e.getNodeA() * nextSize + e.getNodeB()] += g * g;
		return -learnRate * (g / (Math.sqrt(h[l][e.getNodeA() * nextSize + e.getNodeB()]) + epsilon) + lambda * e.getWeight());
	}
	
	@Override
	public double optimizeBias(int l, int i, double[] nextResult, double[] error, Activation activation, int size, int max){
		if(hb == null){
			hb = new double[size][max];
		}
		double g = error[i] * activation.derivative(nextResult[i]);
		hb[l][i] += g * g;
		return -learnRate * (g / (Math.sqrt(hb[l][i]) + epsilon));
	}
}
