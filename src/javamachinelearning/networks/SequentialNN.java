package javamachinelearning.networks;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import javamachinelearning.layers.Layer;
import javamachinelearning.layers.ParamsLayer;
import javamachinelearning.layers.recurrent.RecurrentLayer;
import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.Utils;

public class SequentialNN implements NeuralNetwork, SupervisedNeuralNetwork{
	private ArrayList<Layer> layers = new ArrayList<Layer>();
	private int[] inputShape;
	
	public SequentialNN(int... inputShape){
		if(inputShape.length > 1)
			this.inputShape = inputShape;
		else
			this.inputShape = new int[]{1, inputShape[0]};
	}
	
	public int size(){
		return layers.size();
	}
	
	public Layer layer(int idx){
		return layers.get(idx);
	}
	
	public void add(Layer l){
		l.init(layers.isEmpty() ? inputShape : layers.get(layers.size() - 1).outputShape());
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
	
	// predict using a specific number of time steps
	public Tensor predict(Tensor input, int timeSteps){
		for(int i = 0; i < layers.size(); i++){
			if(layers.get(i) instanceof RecurrentLayer)
				input = ((RecurrentLayer)layers.get(i)).forwardPropagate(input, timeSteps, false);
			else
				input = layers.get(i).forwardPropagate(input, false);
		}
		return input;
	}
	
	// predictTrain should only be used for training!
	// it saves the outputs for each layer
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
		return layers.get(layers.size() - 1).outputShape();
	}
	
	@Override
	public void train(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle){
		train(input, target, epochs, batchSize, loss, optimizer, regularizer, shuffle, false, null);
	}
	
	@Override
	public void train(Tensor[] input, Tensor[] target, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle, boolean verbose){
		train(input, target, epochs, batchSize, loss, optimizer, regularizer, shuffle, verbose, null);
	}
	
	@Override
	public void train(Tensor[] inputParam, Tensor[] targetParam, int epochs, int batchSize, Loss loss, Optimizer optimizer, Regularizer regularizer, boolean shuffle, boolean verbose, ProgressFunction f){
		// make sure shuffling does not affect the input data
		Tensor[] input = inputParam.clone();
		Tensor[] target = targetParam.clone();
		
		for(int i = 0; i < epochs; i++){
			double totalLoss = 0.0;
			
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(Utils.makeStr('=', 75));
				System.out.println("Epoch " + i);
				System.out.println();
			}
			
			if(shuffle)
				Utils.shuffle(input, target);
			
			for(int j = 0; j < input.length; j++){
				Tensor[] res = predictTrain(input[j]);
				
				totalLoss += loss.loss(res[res.length - 1], target[j]).reduce(0, (a, b) -> a + b);
				
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
			
			if(i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0){
				if(verbose){
					System.out.println();
				}else{
					System.out.print("Epoch " + i + "\t");
				}
				System.out.println("Average Loss: " + Utils.format(totalLoss / input.length));
			}
			if(verbose && (i == epochs - 1 || (epochs < 10 ? 0 : (i % (epochs / 10))) == 0)){
				System.out.println(Utils.makeStr('=', 75));
			}
			
			if(f != null)
				f.apply(i, totalLoss / input.length);
		}
	}
	
	public void backPropagate(Tensor[] result, Tensor error){
		for(int i = layers.size() - 1; i >= 0; i--){
			error = layers.get(i).backPropagate(result[i], result[i + 1], error);
		}
	}
	
	// resets the saved states of stateful recurrent layers
	public void resetStates(){
		for(int i = 0; i < layers.size(); i++){
			if(layers.get(i) instanceof RecurrentLayer){
				((RecurrentLayer)layers.get(i)).resetState();
			}
		}
	}
	
	@Override
	public String toString(){
		StringBuilder b = new StringBuilder();
		b.append("Sequential Neural Network\n");
		b.append(Utils.makeStr('-', 75) + "\n");
		for(int i = 0; i < layers.size(); i++){
			b.append("\n" + layers.get(i).toString());
			b.append("\n\n" + Utils.makeStr('-', 75) + "\n");
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
