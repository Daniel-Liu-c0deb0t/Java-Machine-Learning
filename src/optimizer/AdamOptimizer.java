package optimizer;

import edge.Edge;
import utils.Activation;

public class AdamOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	private double beta1;
	private double beta2;
	
	private double[][] m = null;
	private double[][] v = null;
	private double[][] mb = null;
	private double[][] vb = null;
	private double beta3;
	private double beta4;
	
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
	public double optimizeWeight(int l, Edge e, double[] prevResult, double[] nextResult, double[] error, double lambda, double weightSum, Activation activation, int size, int max, int nextSize){
		if(m == null || v == null){
			m = new double[size][max];
			v = new double[size][max];
			if(mb == null || vb == null){
				beta3 = beta1;
				beta4 = beta2;
			}
		}
		double g = (error[e.getNodeB()] + lambda * weightSum) * prevResult[e.getNodeA()] * activation.derivative(nextResult[e.getNodeB()]);
		m[l][e.getNodeA() * nextSize + e.getNodeB()] = (beta1 * m[l][e.getNodeA() * nextSize + e.getNodeB()] + (1 - beta1) * g);
		v[l][e.getNodeA() * nextSize + e.getNodeB()] = (beta2 * v[l][e.getNodeA() * nextSize + e.getNodeB()] + (1 - beta2) * g * g);
		return -learnRate * ((m[l][e.getNodeA() * nextSize + e.getNodeB()] / (1 - beta3)) / (Math.sqrt(v[l][e.getNodeA() * nextSize + e.getNodeB()] / (1 - beta4)) + epsilon) + lambda * e.getWeight());
	}
	
	@Override
	public double optimizeBias(int l, int i, double[] nextResult, double[] error, Activation activation, int size, int max){
		if(mb == null || vb == null){
			mb = new double[size][max];
			vb = new double[size][max];
			if(m == null || v == null){
				beta3 = beta1;
				beta4 = beta2;
			}
		}
		double g = error[i] * activation.derivative(nextResult[i]);
		mb[l][i] = (beta1 * mb[l][i] + (1 - beta1) * g);
		vb[l][i] = (beta2 * vb[l][i] + (1 - beta2) * g * g);
		return -learnRate * (mb[l][i] / (1 - beta3)) / (Math.sqrt(vb[l][i] / (1 - beta4)) + epsilon);
	}
}
