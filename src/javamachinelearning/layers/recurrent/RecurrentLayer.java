package javamachinelearning.layers.recurrent;

import java.nio.ByteBuffer;

import javamachinelearning.layers.ParamsLayer;
import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.TensorUtils;

public class RecurrentLayer implements ParamsLayer{
	private RecurrentCell cell;
	
	private int numTimeSteps;
	private int numOutputs;
	
	private Tensor[] states;
	private int changeCount;
	
	public RecurrentLayer(int numTimeSteps, int numOutputs, RecurrentCell cell){
		this.numTimeSteps = numTimeSteps;
		this.numOutputs = numOutputs;
		this.cell = cell;
	}
	
	public RecurrentLayer(int numTimeSteps, RecurrentCell cell){
		this.numTimeSteps = numTimeSteps;
		this.numOutputs = numTimeSteps;
		this.cell = cell;
	}
	
	@Override
	public int[] nextShape(){
		return new int[]{numOutputs, cell.nextSize()[1]};
	}
	
	@Override
	public int[] prevShape(){
		return new int[]{numTimeSteps, cell.prevSize()[1]};
	}

	@Override
	public void init(int[] prevSize){
		states = new Tensor[numTimeSteps];
		
		// prevSize[1] = size of input 1D tensor
		cell.init(prevSize[1], numTimeSteps);
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
		Tensor[] outputs = new Tensor[numOutputs];
		int idx = 0;
		
		// the same recurrent cell is used across multiple time steps!
		// data is fed into the cell repeatedly
		for(int i = 0; i < numTimeSteps; i++){
			Tensor inTensor = i < input.shape()[0] ?
					input.get(i) : new Tensor(cell.prevSize(), false);
			
			Tensor prevState = i == 0 ?
					new Tensor(cell.prevSize(), false) : states[i - 1];
			
			states[i] = cell.forwardPropagate(i, inTensor, prevState, training);
			
			// only output the last few cells
			if(i >= numTimeSteps - numOutputs){
				outputs[idx] = states[i];
				idx++;
			}
		}
		
		return TensorUtils.stack(outputs);
	}
	
	@Override
	public Tensor backPropagate(Tensor input, Tensor output, Tensor nextLayerError){
		Tensor[] prevLayerError = new Tensor[numTimeSteps];
		Tensor nextCellError = new Tensor(cell.nextSize(), false);
		
		for(int i = numTimeSteps - 1; i >= 0; i--){
			Tensor inTensor = i < input.shape()[0] ?
					input.get(i) : new Tensor(cell.prevSize(), false);
			
			Tensor prevState = i == 0 ?
					new Tensor(cell.prevSize(), false) : states[i - 1];
			
			// accumulate the error gradient from the next layer and the next cell
			int idx = i - (numTimeSteps - numOutputs);
			Tensor totalError = (i >= numTimeSteps - numOutputs) ?
					nextCellError.add(nextLayerError.get(idx)) : nextCellError;
			
			Tensor[] arr = cell.backPropagate(i, inTensor, prevState, totalError);
			
			prevLayerError[i] = i < input.shape()[0] ?
					arr[0] : new Tensor(cell.prevSize(), false);
			
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
}
