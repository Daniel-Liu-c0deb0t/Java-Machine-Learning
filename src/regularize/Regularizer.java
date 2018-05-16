package regularize;

import utils.Tensor;

public interface Regularizer{
	// no need to actually compute the regularization
	public Tensor derivative(Tensor w);
}
