package javamachinelearning.regularizers;

import javamachinelearning.utils.Tensor;

public interface Regularizer{
	// no need to actually compute the regularization
	public Tensor derivative(Tensor w);
}
