package javamachinelearning.optimizers;

import javamachinelearning.utils.Tensor;

public interface Optimizer{
	// called every training iteration, after optimizing weights/biases
	public void update();
	
	// how many extra parameters per weight/bias
	public int extraParams();
	
	// some optimizers might modify the extra params!
	public Tensor optimize(Tensor grads, Tensor[] params);
}
