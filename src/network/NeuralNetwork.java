package network;

import layer.Layer;
import optimizer.Optimizer;
import regularize.Regularizer;
import utils.Tensor;

public interface NeuralNetwork{
	public int size();
	public Layer layer(int idx);
	public void add(Layer l);
	public Tensor[] predict(Tensor[] input);
	public Tensor predict(Tensor input);
	public Tensor[] predictFull(Tensor input);
	public void backPropagate(Tensor[] result, Tensor error, Optimizer optimizer, Regularizer regularizer);
	public int[] inputSize();
	public int[] outputSize();
	public void saveToFile(String path);
	public void loadFromFile(String path);
}
