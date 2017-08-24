package network;

import java.util.ArrayList;

import layer.Layer;

public interface NeuralNetwork{
	public int size();
	public ArrayList<Layer> layers();
	public void add(Layer l);
	public void add(Layer l, double[][] weights, double[] bias);
	public double[][] predict(double[][] input);
	public double[] predict(double[] input);
	public double[][] predictFull(double[] input);
	public int getInputSize();
	public int getOutputSize();
}
