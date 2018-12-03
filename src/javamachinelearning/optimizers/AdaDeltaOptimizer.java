package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public class AdaDeltaOptimizer implements Optimizer{
	private static final double epsilon = 0.00000001;
	private double rho;
	private Tensor learnRate;
	
	public AdaDeltaOptimizer() {
		this.rho = 0.95;
		this.learnRate.add(0.001);
	}
	
	public AdaDeltaOptimizer(double learnRate) {
		this.rho = 0.95;
		this.learnRate.add(learnRate);
	}
	
	@Override
	public void update() {
		// notihing to do
	}

	@Override
	public int extraParams() {
		return 0;
	}

	@Override
	public Tensor optimize(Tensor grads, Tensor[] params) {
		params[0] = params[0].mul(rho).add((grads.mul(grads)).mul(1.0-rho));
		Tensor t = grads.mul(learnRate.add(epsilon)).div(params[0].map(x -> Math.sqrt(x)).add(epsilon));
		learnRate = t.mul(t).mul(1.0-rho).add(params[1]).mul(rho);
		return t;
	}
}
