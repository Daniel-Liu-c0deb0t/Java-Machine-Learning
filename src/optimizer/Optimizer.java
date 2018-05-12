package optimizer;

import utils.Tensor;

public interface Optimizer{
	// should be called before optimizer is used
	public void init(int[][] weightShapes, int[][] biasShapes);
	// called every training iteration, after optimizing weights/bias
	public void update();
	
	public Tensor optimizeWeight(Tensor grads, int l);
	public Tensor optimizeBias(Tensor grads, int l);
}
