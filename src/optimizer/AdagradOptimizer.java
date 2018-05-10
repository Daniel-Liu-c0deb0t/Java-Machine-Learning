package optimizer;

import edge.Edge;
import utils.Activation;
import utils.Tensor;

public class AdagradOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double learnRate;
	
	private Tensor[] hWeight;
	private Tensor[] hBias;
	
	public AdagradOptimizer(){
		this.learnRate = 0.01;
	}
	
	public AdagradOptimizer(double learnRate){
		this.learnRate = learnRate;
	}
	
	@Override
	public void init(int[][] weightShapes, int[][] biasShapes){
		hWeight = new Tensor[weightShapes.length];
		hBias = new Tensor[weightShapes.length];
		
		for(int i = 0; i < weightShapes.length; i++){
			hWeight[i] = new Tensor(weightShapes[i], false);
			hBias[i] = new Tensor(biasShapes[i], false);
		}
	}
	
	@Override
	public void update(){
		// nothing to do
	}
	
	@Override
	public Tensor optimizeWeight(Tensor grads, int l){
		hWeight[l] = hWeight[l].add(grads.mul(grads));
		return grads.mul(-learnRate).div(hWeight[l].map(x -> Math.sqrt(x)).add(epsilon));
	}
	
	@Override
	public Tensor optimizeBias(Tensor grads, int l){
		hBias[l] = hBias[l].add(grads.mul(grads));
		return grads.mul(-learnRate).div(hBias[l].map(x -> Math.sqrt(x)).add(epsilon));
	}
}
