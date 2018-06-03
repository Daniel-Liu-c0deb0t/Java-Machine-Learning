package javamachinelearning.networks;

import javamachinelearning.utils.Tensor;

public interface NeuralNetwork{
	public Tensor[] predict(Tensor[] input);
	public Tensor predict(Tensor input);
	public int[] inputShape();
	public int[] outputShape();
	public void saveToFile(String path);
	public void loadFromFile(String path);
}
