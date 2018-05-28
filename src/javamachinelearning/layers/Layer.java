package javamachinelearning.layers;

import javamachinelearning.utils.Tensor;

public interface Layer{
	public int[] nextShape();
	public int[] prevShape();
	public void init(int[] prevShape);
	public Tensor forwardPropagate(Tensor input, boolean training);
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error);
}
