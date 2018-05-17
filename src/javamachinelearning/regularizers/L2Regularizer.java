package javamachinelearning.regularizers;

import javamachinelearning.utils.Tensor;

public class L2Regularizer implements Regularizer{
	private double lambda;
	
	public L2Regularizer(){
		this.lambda = 0.01;
	}
	
	public L2Regularizer(double lambda){
		this.lambda = lambda;
	}
	
	@Override
	public Tensor derivative(Tensor w){
		return w.mul(lambda);
	}
}
