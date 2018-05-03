package network;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Random;

import layer.Layer;
import optimizer.Update;
import optimizer.Optimizer;
import optimizer.SGDOptimizer;
import utils.Loss;
import utils.UtilMethods;

public class SimpleNeuralNetwork implements NeuralNetwork, SupervisedNeuralNetwork{
	private ArrayList<Layer> layers = new ArrayList<Layer>();
	private int inputSize;
	
	public SimpleNeuralNetwork(int inputSize){
		this.inputSize = inputSize;
	}
	
	@Override
	public int size(){
		return layers.size();
	}

	@Override
	public ArrayList<Layer> layers(){
		return layers;
	}

	@Override
	public void add(Layer l){
		l.init(layers.isEmpty() ? inputSize : layers.get(layers.size() - 1).nextSize());
		layers.add(l);
	}
	
	@Override
	public void add(Layer l, double[][] weights, double[] bias){
		l.init(layers.isEmpty() ? inputSize : layers.get(layers.size() - 1).nextSize(), weights, bias);
		layers.add(l);
	}

	@Override
	public double[][] predict(double[][] input){
		double[][] result = new double[input.length][layers.get(layers.size() - 1).nextSize()];
		
		for(int i = 0; i < input.length; i++){
			result[i] = predict(input[i]);
		}
		
		return result;
	}
	
	@Override
	public double[] predict(double[] input){
		for(int i = 0; i < layers.size(); i++){
			input = layers.get(i).forwardPropagate(input);
		}
		return input;
	}
	
	@Override
	public double[][] predictFull(double[] input){
		int max = input.length;
		for(int i = 0; i < layers.size(); i++){
			max = Math.max(max, layers.get(i).nextSize());
		}
		
		double[][] result = new double[layers.size() + 1][max];
		for(int i = 0; i < input.length; i++){
			result[0][i] = input[i];
		}
		for(int i = 1; i < layers.size() + 1; i++){
			input = layers.get(i - 1).forwardPropagate(input);
			for(int j = 0; j < input.length; j++){
				result[i][j] = input[j];
			}
		}
		return result;
	}
	
	@Override
	public int getInputSize(){
		return inputSize;
	}
	
	@Override
	public int getOutputSize(){
		return layers.get(layers.size() - 1).nextSize();
	}
	
	@Override
	public void fit(double[][] input, double[][] target, boolean verbose, boolean printNet){
		fit(input, target, 1000, 1, verbose, printNet);
	}
	
	@Override
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, boolean verbose, boolean printNet){
		fit(input, target, epochs, batchSize, Loss.squared, new SGDOptimizer(), verbose, printNet);
	}
	
	@Override
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, boolean verbose, boolean printNet){
		fit(input, target, epochs, batchSize, loss, optimizer, 0.0, verbose, printNet);
	}
	
	@Override
	public void fit(double[][] input, double[][] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, double regLambda, boolean verbose, boolean printNet){
		double weightSum = 0.0;
		int max = 0;
		int max2 = 0;
		for(int i = 0; i < layers.size(); i++){
			for(int j = 0; j < layers.get(i).edges().length; j++){
				weightSum += regLambda * layers.get(i).edges()[j].getWeight();
			}
			max = Math.max(max, layers.get(i).edges().length);
			max2 = Math.max(max2, layers.get(i).nextSize());
		}
		
		for(int i = 0; i < epochs; i++){
			double totalLoss = 0.0;
			
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(UtilMethods.makeStr('=', 30));
				System.out.println("Epoch " + UtilMethods.format(i) + ":");
				System.out.println();
				System.out.println(UtilMethods.makeStr('-', 5) + " Before " + UtilMethods.makeStr('-', 5));
				if(printNet)
					UtilMethods.printNN(this);
				System.out.println(UtilMethods.makeStr('-', 18));
				System.out.println();
			}
			
			Random r = new Random();
			
			for(int j = 0; j < input.length; j++){
				double[][] result = predictFull(input[j]);
				
				for(int k = 0; k < layers.size(); k++){
					for(int l = 0; l < layers.get(k).nextSize(); l++){
						if(r.nextDouble() < layers.get(k).getDropout()){
							result[k + 1][l] = 0.0;
						}else{
							result[k + 1][l] /= 1.0 - layers.get(k).getDropout();
						}
					}
				}
				
				totalLoss += loss.loss(result[result.length - 1], target[j]);
				
				if(verbose && ((i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0) && (input.length < 10 ? 0 : (j % (input.length / 10))) == 0)){
					System.out.print("Input: ");
					UtilMethods.printArray(input[j]);
					System.out.print("Output: ");
					UtilMethods.printArray(result[result.length - 1], getOutputSize());
					System.out.print("Target: ");
					UtilMethods.printArray(target[j]);
					System.out.println();
				}
				
				// add regularization's derivative to the loss function's derivative
				double[] lossDerivative = loss.derivative(result[result.length - 1], target[j]);
				for(int k = 0; k < lossDerivative.length; k++){
					lossDerivative[k] += weightSum;
				}
				
				backPropagate(result, lossDerivative, regLambda, optimizer, max, max2);
				
				if(j + 1 % batchSize == 0 || j == input.length - 1){
					weightSum = 0.0;
					for(int k = 0; k < layers.size(); k++){
						layers.get(k).update();
						for(int l = 0; l < layers.get(k).edges().length; l++){
							weightSum += regLambda * layers.get(k).edges()[l].getWeight();
						}
					}
				}
			}
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(UtilMethods.makeStr('-', 5) + " After " + UtilMethods.makeStr('-', 6));
				if(printNet)
					UtilMethods.printNN(this);
				System.out.println(UtilMethods.makeStr('-', 18));
			}
			if(i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0){
				if(verbose){
					System.out.println();
				}
				System.out.println("Loss: " + UtilMethods.format(totalLoss / input.length));
			}
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(UtilMethods.makeStr('=', 30));
			}
		}
	}
	
	@Override
	public void backPropagate(double[][] result, double[] error, double regLambda, Optimizer optimizer, int max, int max2){
		for(int i = size() - 1; i >= 0; i--){
			error = layers.get(i).backPropagate(result[i], result[i + 1], error, regLambda, optimizer, i, layers.size(), max, max2);
		}
	}
	
	@Override
	public void saveToFile(String path){
		int totalLayerSize = 0;
		for(int i = 0; i < layers.size(); i++){
			totalLayerSize += layers.get(i).byteSize();
		}
		ByteBuffer bb = ByteBuffer.allocate(totalLayerSize);
		for(int i = 0; i < layers.size(); i++){
			bb.put(layers.get(i).toBytes());
		}
		bb.flip();
		try{
			Files.write(Paths.get(path), bb.array());
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	@Override
	public void loadFromFile(String path){
		byte[] bytes = null;
		try{
			bytes = Files.readAllBytes(Paths.get(path));
		}catch(Exception e){
			e.printStackTrace();
		}
		ByteBuffer bb = ByteBuffer.wrap(bytes);
		for(int i = 0; i < layers.size(); i++){
			for(int j = 0; j < layers.get(i).edges().length; j++){
				layers.get(i).edges()[j].setWeight(bb.getDouble());
			}
			for(int j = 0; j < layers.get(i).getBias().length; j++){
				layers.get(i).getBias()[j] = bb.getDouble();
			}
		}
	}
}
