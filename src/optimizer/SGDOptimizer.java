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
	public Deltas optimize(NeuralNetwork nn, double[][] result, double[] error, double lambda, double weightSum){
		int max = 0;
		int max2 = 0;
		for(int i = 0; i < nn.size(); i++){
			max = Math.max(max, nn.layers().get(i).edges().length);
			max2 = Math.max(max, nn.layers().get(i).nextSize());
		}
		
		double[] error2 = new double[error.length];
		for(int i = 0; i < error.length; i++){
			error2[i] = error[i];
		}
		double[][] delta = new double[nn.size()][max];
		double[][] biasDelta = new double[nn.size()][max2];
		for(int i = nn.size() - 1; i >= 0; i--){
			Layer l = nn.layers().get(i);
			double[] newError = new double[l.prevSize()];
			double[] newError2 = new double[l.prevSize()];
			for(int j = 0; j < l.edges().length; j++){
				Edge e = l.edges()[j];
				double g = (error[e.getNodeB()] + lambda * weightSum) * result[i][e.getNodeA()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], null);
				g += lambda * e.getWeight();
				delta[i][j] = -learnRate * g;
				newError[e.getNodeA()] += e.getWeight() * (error[e.getNodeB()] + lambda * weightSum) * l.getActivationP().activate(result[i + 1][e.getNodeB()], null);
				newError2[e.getNodeA()] += e.getWeight() * error2[e.getNodeB()] * l.getActivationP().activate(result[i + 1][e.getNodeB()], null);
			}
			for(int j = 0; j < l.nextSize(); j++){
				double g = error2[j] * l.getActivationP().activate(result[i + 1][j], null);
				biasDelta[i][j] = -learnRate * g;
			}
			error = newError;
			error2 = newError2;
		}
		return new Deltas(delta, biasDelta);
	}
}
