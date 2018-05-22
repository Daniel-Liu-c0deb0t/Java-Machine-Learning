package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class MomentumOptimizer implements Optimizer{
	private double learnRate;
	private double mu; // friction to decay momentum
	private boolean useNesterov;
	
	private Tensor[] vWeight;
	private Tensor[] vBias;
	
	public MomentumOptimizer(){
		this.learnRate = 0.1;
		this.mu = 0.9;
		this.useNesterov = false;
	}
	
	public MomentumOptimizer(double learnRate){
		this.learnRate = learnRate;
		this.mu = 0.9;
		this.useNesterov = false;
	}
	
	public MomentumOptimizer(double learnRate, boolean useNesterov){
		this.learnRate = learnRate;
		this.mu = 0.9;
		this.useNesterov = useNesterov;
	}
	
	public MomentumOptimizer(double learnRate, double mu, boolean useNesterov){
		this.learnRate = learnRate;
		this.mu = mu;
		this.useNesterov = useNesterov;
	}
	
	@Override
	public void init(int[][] weightShapes, int[][] biasShapes){
		vWeight = new Tensor[weightShapes.length];
		vBias = new Tensor[weightShapes.length];
		
		for(int i = 0; i < weightShapes.length; i++){
			if(weightShapes[i] != null)
				vWeight[i] = new Tensor(weightShapes[i], false);
			if(biasShapes[i] != null)
				vBias[i] = new Tensor(biasShapes[i], false);
		}
	}
	
	@Override
	public void update(){
		// nothing to do
	}
	
	@Override
	public Tensor optimizeWeight(Tensor grads, int l){
		if(useNesterov){
			Tensor prev = vWeight[l];
			vWeight[l] = vWeight[l].mul(mu).sub(grads.mul(learnRate));
			return prev.mul(mu).sub(vWeight[l].mul(1.0 + mu)); // is negated
		}else{
			vWeight[l] = vWeight[l].mul(mu).sub(grads.mul(learnRate));
			return vWeight[l].mul(-1.0); // is negated
		}
	}
	
	@Override
	public Tensor optimizeBias(Tensor grads, int l){
		if(useNesterov){
			Tensor prev = vBias[l];
			vBias[l] = vBias[l].mul(mu).sub(grads.mul(learnRate));
			return prev.mul(mu).sub(vBias[l].mul(1.0 + mu)); // is negated
		}else{
			vBias[l] = vBias[l].mul(mu).sub(grads.mul(learnRate));
			return vBias[l].mul(-1.0); // is negated
		}
	}
}
