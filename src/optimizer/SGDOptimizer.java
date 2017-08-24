package optimizer;

import edge.Edge;
import layer.Layer;
import network.NeuralNetwork;

public class SGDOptimizer implements Optimizer{
	private double learnRate;
	
	public SGDOptimizer(){
		this.learnRate = 0.01;
	}
	
	public SGDOptimizer(double learnRate){
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
		
		double[][] delta = new double[nn.size()][max];
		double[][] biasDelta = new double[nn.size()][max2];
		for(int i = nn.size() - 1; i >= 0; i--){
			Layer l = nn.layers().get(i);
			double[] newError = new double[l.prevSize()];
			for(int j = 0; j < l.edges().length; j++){
				Edge e = l.edges()[j];
				double g = error[e.getNodeB()] * result[i][e.getNodeA()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], i == nn.size() - 1 ? new double[]{target[e.getNodeB()]} : null);
				g += lambda * e.getWeight();
				delta[i][j] = -learnRate * g;
				newError[e.getNodeA()] += e.getWeight() * error[e.getNodeB()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], i == nn.size() - 1 ? new double[]{target[e.getNodeB()]} : null);
			}
			for(int j = 0; j < l.nextSize(); j++){
				double g = error[j] * l.getActivationP().activate(result[i + 1][j], i == nn.size() - 1 ? new double[]{target[j]} : null);
				biasDelta[i][j] = -learnRate * g;
			}
			error = newError;
		}
		return new Deltas(delta, biasDelta);
	}
}
