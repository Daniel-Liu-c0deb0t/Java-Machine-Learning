package network;

import java.util.ArrayList;

import layer.Layer;
import optimizer.Optimizer;

public interface NeuralNetwork{
	public int size();
	public ArrayList<Layer> layers();
	public void add(Layer l);
	public void add(Layer l, double[][] weights, double[] bias);
	public double[][] predict(double[][] input);
	public double[] predict(double[] input);
	public double[][] predictFull(double[] input);
	public void backPropagate(double[][] result, double[] error, double regLambda, Optimizer optimizer, int max, int max2);
	public int getInputSize();
	public int getOutputSize();
	public void saveToFile(String path);
	public void loadFromFile(String path);
}
