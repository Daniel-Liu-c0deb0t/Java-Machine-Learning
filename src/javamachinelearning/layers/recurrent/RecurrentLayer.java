package javamachinelearning.layers.recurrent;

import java.nio.ByteBuffer;
import java.util.Arrays;

import javamachinelearning.layers.ParamsLayer;
import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.TensorUtils;

public class RecurrentLayer implements ParamsLayer{
	private RecurrentCell cell;
	private boolean statefulTrain;
	private boolean statefulTest;
	
	private int numTimeSteps;
	private int numOutputs;
	private boolean outputAll;
	
	private Tensor[] states;
	private int changeCount;
	
	// state from the previous forward propagation of this layer
	// allows the recurrent cells to continue where it left off before
	private Tensor layerPrevState;
	// save the previous state before it is updated, for backpropagation
	private Tensor layerPrevStateTemp;
	
	public RecurrentLayer(int numTimeSteps, int numOutputs, RecurrentCell cell, boolean statefulTrain, boolean statefulTest){
		this.numTimeSteps = numTimeSteps;
		this.numOutputs = numOutputs;
		this.cell = cell;
		this.statefulTrain = statefulTrain;
		this.statefulTest = statefulTest;
		this.outputAll = false;
	}
	
	public RecurrentLayer(int numTimeSteps, RecurrentCell cell, boolean statefulTrain, boolean statefulTest){
		this.numTimeSteps = numTimeSteps;
		this.numOutputs = numTimeSteps;
		this.cell = cell;
		this.statefulTrain = statefulTrain;
		this.statefulTest = statefulTest;
		this.outputAll = true;
	}
	
	public RecurrentLayer(int numTimeSteps, RecurrentCell cell, boolean stateful){
		this.numTimeSteps = numTimeSteps;
		this.numOutputs = numTimeSteps;
		this.cell = cell;
		this.statefulTrain = stateful;
		this.statefulTest = stateful;
		this.outputAll = true;
	}
	
	@Override
	public int[] outputShape(){
		return new int[]{numOutputs, cell.outputShape()[1]};
	}
	
	@Override
	public int[] inputShape(){
		return new int[]{numTimeSteps, cell.inputShape()[1]};
	}

	@Override
	public void init(int[] inputShape){
		states = new Tensor[numTimeSteps];
		
		// inputShape[1] = size of input 1D tensor
		cell.init(inputShape[1], numTimeSteps);
	}
	
	@Override
	public ParamsLayer noBias(){
		cell.noBias();
		return this;
	}
	
	public RecurrentCell cell(){
		return cell;
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		return forwardPropagate(input, numTimeSteps, training);
	}
	
	// more general method that allows the number of times the cell is propagated through to vary
	public Tensor forwardPropagate(Tensor input, int timeSteps, boolean training){
		int outputCount = outputAll ? timeSteps : Math.min(numOutputs, timeSteps);
		boolean stateful = (training && statefulTrain) || (!training && statefulTest);
		Tensor[] outputs = new Tensor[outputCount];
		int idx = 0;
		
		// the same recurrent cell is used across multiple time steps!
		// data is fed into the cell repeatedly
		for(int i = 0; i < timeSteps; i++){
			Tensor inTensor = i < input.shape()[0] ?
					input.get(i) : new Tensor(cell.inputShape(), false);
			
			Tensor prevState = i == 0 ?
					(stateful && layerPrevState != null ? layerPrevState :
						new Tensor(cell.inputShape(), false)) : states[i - 1];
			
			states[i] = cell.forwardPropagate(i, inTensor, prevState, training);
			
			// only output the last few cells
			if(i >= timeSteps - outputCount){
				outputs[idx] = states[i];
				idx++;
			}
		}
		
		// save last state for next time this layer is forward propagated, if necessary
		if(stateful){
			layerPrevStateTemp = layerPrevState;
			layerPrevState = states[timeSteps - 1];
		}
		
		return TensorUtils.stack(outputs);
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor nextLayerError){
		Tensor[] prevLayerError = new Tensor[numTimeSteps];
		Tensor nextCellError = new Tensor(cell.outputShape(), false);
		
		for(int i = numTimeSteps - 1; i >= 0; i--){
			Tensor inTensor = i < input.shape()[0] ?
					input.get(i) : new Tensor(cell.inputShape(), false);
			
			Tensor prevState = i == 0 ?
					(statefulTrain && layerPrevStateTemp != null ? layerPrevStateTemp :
						new Tensor(cell.inputShape(), false)) : states[i - 1];
			
			// accumulate the error gradient from the next layer and the next cell
			int idx = i - (numTimeSteps - numOutputs);
			Tensor totalError = (i >= 0) ? nextCellError.add(nextLayerError.get(idx)) : nextCellError;
			
			Tensor[] arr = cell.backPropagate(i, inTensor, prevState, totalError);
			
			prevLayerError[i] = i < input.shape()[0] ?
					arr[0] : new Tensor(cell.inputShape(), false);
			
			nextCellError = arr[1];
		}
		
		changeCount++;
		
		return TensorUtils.stack(prevLayerError);
	}
	
	@Override
	public void update(Optimizer optimizer, Regularizer regularizer){
		cell.update(optimizer, regularizer, changeCount);
		changeCount = 0;
	}
	
	// reset the previous state that is saved if this model is stateful
	public void resetState(){
		layerPrevState = null;
		layerPrevStateTemp = null;
	}
	
	@Override
	public int byteSize(){
		return cell.byteSize();
	}
	
	@Override
	public ByteBuffer bytes(){
		ByteBuffer bb = ByteBuffer.allocate(byteSize());
		bb.put(cell.bytes());
		bb.flip();
		return bb;
	}
	
	@Override
	public void readBytes(ByteBuffer bb){
		cell.readBytes(bb);
	}
	
	@Override
	public String toString(){
		return "Recurrent\tCell: " + cell.toString() + "\tInput Shape: " + Arrays.toString(inputShape()) + "\tOutput Shape: " + Arrays.toString(outputShape());
	}
}
