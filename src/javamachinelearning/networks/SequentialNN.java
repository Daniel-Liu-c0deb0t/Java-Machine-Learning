package javamachinelearning.networks;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import javamachinelearning.layers.Layer;
import javamachinelearning.layers.ParamsLayer;
import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class SequentialNN implements NeuralNetwork, SupervisedNeuralNetwork{
	private ArrayList<Layer> layers = new ArrayList<Layer>();
	private int[] inputShape;
	
	public SequentialNN(int... inputShape){
		this.inputShape = inputShape;
	}
	
	@Override
	public int size(){
		return layers.size();
	}

	@Override
	public Layer layer(int idx){
		return layers.get(idx);
	}

	@Override
	public void add(Layer l){
		l.init(layers.isEmpty() ? inputShape : layers.get(layers.size() - 1).nextShape());
		layers.add(l);
	}

	@Override
	public Tensor[] predict(Tensor[] input){
		Tensor[] res = new Tensor[input.length];
		
		for(int i = 0; i < input.length; i++){
			res[i] = predict(input[i]);
		}
		
		return res;
	}
	
	@Override
	public Tensor predict(Tensor input){
		for(int i = 0; i < layers.size(); i++){
			input = layers.get(i).forwardPropagate(input, false);
		}
		return input;
	}
	
	// predictTrain should only be used for training!
	// it saves the outputs for each layer
	@Override
	public Tensor[] predictTrain(Tensor input){
		Tensor[] res = new Tensor[layers.size() + 1];
		res[0] = input;
		for(int i = 1; i < layers.size() + 1; i++){
			input = layers.get(i - 1).forwardPropagate(input, true);
			res[i] = input;
		}
		return res;
	}
	
	@Override
	public int[] inputShape(){
		return inputShape;
	}
	
	@Override
	public int[] outputShape(){
		return layers.get(layers.size() - 1).nextShape();
	}
	
	@Override
	public void train(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle){
		train(input, target, epochs, batchSize, loss, optimizer, regularizer, shuffle, false, false, null);
	}
	
	@Override
	public void train(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle, boolean verbose, boolean printNet){
		train(input, target, epochs, batchSize, loss, optimizer, regularizer, shuffle, verbose, printNet, null);
	}
	
	@Override
	public void train(Tensor[] inputParam, Tensor[] targetParam, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle, boolean verbose, boolean printNet, ProgressFunction f){
		// make sure shuffling does not affect the input data
		Tensor[] input = inputParam.clone();
		Tensor[] target = targetParam.clone();
		
		for(int i = 0; i < epochs; i++){
			double totalLoss = 0.0;
			
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(Utils.makeStr('=', 30));
				System.out.println("Epoch " + i);
				System.out.println();
				System.out.println(Utils.makeStr('-', 5) + " Before " + Utils.makeStr('-', 5));
				if(printNet)
					System.out.println(toString());
				System.out.println(Utils.makeStr('-', 18));
				System.out.println();
			}
			
			if(shuffle)
				Utils.shuffle(input, target);
			
			for(int j = 0; j < input.length; j++){
				Tensor[] res = predictTrain(input[j]);
				
				totalLoss += loss.loss(res[res.length - 1], target[j]);
				
				if(verbose && ((i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0) && (input.length < 10 ? 0 : (j % (input.length / 10))) == 0)){
					System.out.print("Input: ");
					System.out.println(input[j]);
					System.out.print("Output: ");
					System.out.println(res[res.length - 1]);
					System.out.print("Target: ");
					System.out.println(target[j]);
					System.out.println();
				}
				
				// calculate derivative of the loss function and backpropagate
				Tensor lossDerivative = loss.derivative(res[res.length - 1], target[j]);
				backPropagate(res, lossDerivative);
				
				// update weights and biases if batch size is reached
				if(j + 1 % batchSize == 0 || j == input.length - 1){
					for(int k = 0; k < layers.size(); k++){
						if(layers.get(k) instanceof ParamsLayer)
							((ParamsLayer)layers.get(k)).update(optimizer, regularizer);
					}
					
					optimizer.update();
				}
			}
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(Utils.makeStr('-', 5) + " After " + Utils.makeStr('-', 6));
				if(printNet)
					System.out.println(toString());
				System.out.println(Utils.makeStr('-', 18));
			}
			if(i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0){
				if(verbose){
					System.out.println();
				}else{
					System.out.print("Epoch " + i + "\n\t");
				}
				System.out.println("Average loss: " + Utils.format(totalLoss / input.length));
			}
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(Utils.makeStr('=', 30));
			}
			
			if(f != null)
				f.apply(i, totalLoss / input.length);
		}
	}
	
	@Override
	public void backPropagate(Tensor[] result, Tensor error){
		for(int i = layers.size() - 1; i >= 0; i--){
			error = layers.get(i).backPropagate(result[i], result[i + 1], error);
		}
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		for(int i = 0; i < layers.size(); i++){
			b.append(layers.get(i).toString());
			b.append("\n" + Utils.makeStr('=', 50) + "\n");
		}
		return b.toString();
	}
	
	@Override
	public void saveToFile(String path){
		int totalLayerSize = 0;
		for(int i = 0; i < layers.size(); i++){
			if(layers.get(i) instanceof ParamsLayer)
				totalLayerSize += ((ParamsLayer)layers.get(i)).byteSize();
		}
		ByteBuffer bb = ByteBuffer.allocate(totalLayerSize);
		for(int i = 0; i < layers.size(); i++){
			if(layers.get(i) instanceof ParamsLayer)
				bb.put(((ParamsLayer)layers.get(i)).bytes());
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
			if(layers.get(i) instanceof ParamsLayer)
				((ParamsLayer)layers.get(i)).readBytes(bb);
		}
	}
}
