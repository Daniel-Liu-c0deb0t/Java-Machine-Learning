package optimizer;

import edge.Edge;
import layer.Layer;
import network.NeuralNetwork;

public class AdamOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	private double beta1;
	private double beta2;
	
	private double[][] m = null;
	private double[][] v = null;
	private double[][] mb = null;
	private double[][] vb = null;
	
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
	public Deltas optimize(NeuralNetwork nn, double[][] result, double[] error, double[] target, double lambda){
		int max = 0;
		int max2 = 0;
		for(int i = 0; i < nn.size(); i++){
			max = Math.max(max, nn.layers().get(i).edges().length);
			max2 = Math.max(max, nn.layers().get(i).nextSize());
		}
		if(m == null || v == null){
			m = new double[nn.size()][max];
			v = new double[nn.size()][max];
			mb = new double[nn.size()][max2];
			vb = new double[nn.size()][max2];
		}
		double[][] delta = new double[nn.size()][max];
		double[][] biasDelta = new double[nn.size()][max2];
		for(int i = nn.size() - 1; i >= 0; i--){
			Layer l = nn.layers().get(i);
			double[] newError = new double[l.prevSize()];
			for(int j = 0; j < l.edges().length; j++){
				Edge e = l.edges()[j];
				double g = error[e.getNodeB()] * result[i][e.getNodeA()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], i == nn.size() - 1 ? new double[]{target[e.getNodeB()]} : null);
				m[i][j] = (beta1 * m[i][j] + (1 - beta1) * g);
				v[i][j] = (beta2 * v[i][j] + (1 - beta2) * g * g);
				g += lambda * e.getWeight();
				delta[i][j] = -learnRate * (m[i][j] / (1 - beta1)) / (Math.sqrt(v[i][j] / (1 - beta2)) + epsilon);
				newError[e.getNodeA()] += e.getWeight() * error[e.getNodeB()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], i == nn.size() - 1 ? new double[]{target[e.getNodeB()]} : null);
			}
			for(int j = 0; j < l.nextSize(); j++){
				double g = error[j] * l.getActivationP().activate(result[i + 1][j], i == nn.size() - 1 ? new double[]{target[j]} : null);
				mb[i][j] = (beta1 * mb[i][j] + (1 - beta1) * g);
				vb[i][j] = (beta2 * vb[i][j] + (1 - beta2) * g * g);
				biasDelta[i][j] = -learnRate * (mb[i][j] / (1 - beta1)) / (Math.sqrt(vb[i][j] / (1 - beta2)) + epsilon);
			}
			error = newError;
		}
		return new Deltas(delta, biasDelta);
	}
}
