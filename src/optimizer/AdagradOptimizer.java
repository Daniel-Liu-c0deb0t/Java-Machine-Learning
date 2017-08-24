package optimizer;

import edge.Edge;
import layer.Layer;
import network.NeuralNetwork;

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
	public Deltas optimize(NeuralNetwork nn, double[][] result, double[] error, double[] target, double lambda){
		int max = 0;
		int max2 = 0;
		for(int i = 0; i < nn.size(); i++){
			max = Math.max(max, nn.layers().get(i).edges().length);
			max2 = Math.max(max, nn.layers().get(i).nextSize());
		}
		if(h == null || hb == null){
			h = new double[nn.size()][max];
			hb = new double[nn.size()][max2];
		}
		double[][] delta = new double[nn.size()][max];
		double[][] biasDelta = new double[nn.size()][max2];
		for(int i = nn.size() - 1; i >= 0; i--){
			Layer l = nn.layers().get(i);
			double[] newError = new double[l.prevSize()];
			for(int j = 0; j < l.edges().length; j++){
				Edge e = l.edges()[j];
				double g = error[e.getNodeB()] * result[i][e.getNodeA()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], i == nn.size() - 1 ? new double[]{target[e.getNodeB()]} : null);
				h[i][j] += g * g;
				g += lambda * e.getWeight();
				delta[i][j] = -learnRate * (g / (Math.sqrt(h[i][j]) + epsilon));
				newError[e.getNodeA()] += e.getWeight() * error[e.getNodeB()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], i == nn.size() - 1 ? new double[]{target[e.getNodeB()]} : null);
			}
			for(int j = 0; j < l.nextSize(); j++){
				double g = error[j] * l.getActivationP().activate(result[i + 1][j], i == nn.size() - 1 ? new double[]{target[j]} : null);
				hb[i][j] += g * g;
				biasDelta[i][j] = -learnRate * (g / (Math.sqrt(hb[i][j]) + epsilon));
			}
			error = newError;
		}
		return new Deltas(delta, biasDelta);
	}
}
