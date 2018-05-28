package javamachinelearning.layers;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.TensorUtils;

public class GRULayer implements RecurrentParamsLayer{
	private GRUCell[] cells;
	
	private Activation activation;
	private Activation gateActivation;
	private int numTotalCells;
	private int numOutputCells;
	private int size;
	
	private Tensor[] states;
	
	public GRULayer(int numTotalCells, int numOutputCells, int size, Activation activation, Activation gateActivation){
		this.numTotalCells = numTotalCells;
		this.numOutputCells = numOutputCells;
		this.size = size;
		this.activation = activation;
		this.gateActivation = gateActivation;
	}
	
	public GRULayer(int numTotalCells, int numOutputCells, int size, Activation activation){
		this.numTotalCells = numTotalCells;
		this.numOutputCells = numOutputCells;
		this.size = size;
		this.activation = activation;
		this.gateActivation = Activation.tanh;
	}
	
	public GRULayer(int numTotalCells, int size, Activation activation){
		this.numTotalCells = numTotalCells;
		this.numOutputCells = numTotalCells;
		this.size = size;
		this.activation = activation;
		this.gateActivation = Activation.tanh;
	}
	
	@Override
	public int[] nextShape(){
		return new int[]{numOutputCells, size};
	}
	
	@Override
	public int[] prevShape(){
		return new int[]{numTotalCells, size};
	}

	@Override
	public void init(int[] prevSize){
		cells = new GRUCell[numTotalCells];
		states = new Tensor[numTotalCells];
		for(int i = 0; i < cells.length; i++){
			cells[i] = new GRUCell(size, gateActivation);
		}
	}
	
	@Override
	public RecurrentParamsLayer noBias(){
		for(int i = 0; i < cells.length; i++){
			cells[i].noBias();
		}
		return this;
	}
	
	@Override
	public RecurrentCell[] cells(){
		return cells;
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, boolean training){
		Tensor[] outputs = new Tensor[numOutputCells];
		int idx = 0;
		
		for(int i = 0; i < cells.length; i++){
			Tensor inTensor = i < input.shape()[0] ?
					input.get(i) : new Tensor(new int[]{size}, false);
			Tensor prevState = i == 0 ?
					new Tensor(new int[]{size}, false) : states[i - 1];
			states[i] = cells[i].forwardPropagate(inTensor, prevState, training);
			
			// only output the last few cells
			if(i >= cells.length - numOutputCells){
				outputs[idx] = activation.activate(states[i]);
				idx++;
			}
		}
		return TensorUtils.stack(outputs);
	}
	
	@Override
	public Tensor backPropagate(Tensor prevRes, Tensor nextRes, Tensor error){
		// TODO: backprop output cells activation
	}
	
	@Override
	public void update(Optimizer optimizer, Regularizer regularizer){
		for(int i = 0; i < cells.length; i++){
			cells[i].update(optimizer, regularizer);
		}
	}
	
	@Override
	public int byteSize(){
		int size = 0;
		for(int i = 0; i < cells.length; i++){
			size += cells[i].byteSize();
		}
		return size;
	}
	
	@Override
	public ByteBuffer bytes(){
		ByteBuffer bb = ByteBuffer.allocate(byteSize());
		for(int i = 0; i < cells.length; i++){
			bb.put(cells[i].bytes());
		}
		bb.flip();
		return bb;
	}
	
	@Override
	public void readBytes(ByteBuffer bb){
		for(int i = 0; i < cells.length; i++){
			cells[i].readBytes(bb);
		}
	}
}
