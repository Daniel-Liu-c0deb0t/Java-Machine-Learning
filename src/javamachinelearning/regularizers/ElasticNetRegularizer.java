package javamachinelearning.regularizers;

import javamachinelearning.utils.Tensor;

public class ElasticNetRegularizer implements Regularizer{
	private double lambdaL1;
	private double lambdaL2;
	
	public ElasticNetRegularizer(){
		this.lambdaL1 = 0.001;
		this.lambdaL2 = 0.001;
	}
	
	public ElasticNetRegularizer(double lambdaL1, double lambdaL2){
		this.lambdaL1 = lambdaL1;
		this.lambdaL2 = lambdaL2;
	}
	
	@Override
	public Tensor derivative(Tensor w){
		return w.map(x -> lambdaL1 * Math.signum(x) + lambdaL2 * x);
	}
}
