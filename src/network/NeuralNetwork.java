package network;

import java.util.ArrayList;

import layer.Layer;
import optimizer.Optimizer;
import utils.Tensor;

public interface NeuralNetwork{
	public int size();
	public ArrayList<Layer> layers();
	public void add(Layer l);
	public void add(Layer l, double[][] weights, double[] bias);
	public Tensor[] predict(Tensor[] input);
	public Tensor predict(Tensor input);
	public Tensor[] predictFull(Tensor input);
	public void backPropagate(Tensor[] result, Tensor error, double regLambda, Optimizer optimizer);
	public int inputSize();
	public int outputSize();
	public void saveToFile(String path);
	public void loadFromFile(String path);
}
