package regularize;

import utils.Tensor;

public class L1Regularizer implements Regularizer{
	private double lambda;
	
	public L1Regularizer(){
		this.lambda = 0.01;
	}
	
	public L1Regularizer(double lambda){
		this.lambda = lambda;
	}
	
	@Override
	public Tensor derivative(Tensor w){
		return w.map(x -> lambda * Math.signum(x));
	}
}
