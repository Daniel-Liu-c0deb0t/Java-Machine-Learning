package javamachinelearning.networks;

import javamachinelearning.layers.Layer;
import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Tensor;

public interface NeuralNetwork{
	public int size();
	public Layer layer(int idx);
	public void add(Layer l);
	public Tensor[] predict(Tensor[] input);
	public Tensor predict(Tensor input);
	public Tensor[] predictTrain(Tensor input);
	public void backPropagate(Tensor[] result, Tensor error, Optimizer optimizer, Regularizer regularizer);
	public int[] inputShape();
	public int[] outputShape();
	public void saveToFile(String path);
	public void loadFromFile(String path);
}
