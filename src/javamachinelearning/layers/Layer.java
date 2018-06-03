package javamachinelearning.layers;

import javamachinelearning.utils.Tensor;

public interface Layer{
	public int[] inputShape();
	public int[] outputShape();
	public void init(int[] inputShape);
	public Tensor forwardPropagate(Tensor input, boolean training);
	public Tensor backPropagate(Tensor input, Tensor output, Tensor error);
}
