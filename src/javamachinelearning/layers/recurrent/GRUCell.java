package javamachinelearning.layers.recurrent;

import java.nio.ByteBuffer;

import javamachinelearning.optimizers.Optimizer;
import javamachinelearning.regularizers.Regularizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;

public class GRUCell implements RecurrentCell{
	private int size;
	private Activation activation;
	
	private Tensor resetW, updateW, memoryW;
	private Tensor resetU, updateU, memoryU;
	private Tensor resetB, updateB, memoryB;
	
	private Tensor gradResetW, gradUpdateW, gradMemoryW;
	private Tensor gradResetU, gradUpdateU, gradMemoryU;
	private Tensor gradResetB, gradUpdateB, gradMemoryB;
	
	private Tensor[] resetWParams, updateWParams, memoryWParams;
	private Tensor[] resetUParams, updateUParams, memoryUParams;
	private Tensor[] resetBParams, updateBParams, memoryBParams;
	
	// cached values for backpropagation
	private Tensor[] reset, update, memory;
	
	private boolean useBias = true;
	
	public GRUCell(Activation activation){
		this.activation = activation;
	}
	
	public GRUCell(){
		this(Activation.tanh);
	}
	
	@Override
	public void noBias(){
		resetB = null;
		updateB = null;
		memoryB = null;
		gradResetB = null;
		gradUpdateB = null;
		gradMemoryB = null;
		useBias = false;
	}
	
	@Override
	public int[] nextSize(){
		return new int[]{1, size};
	}
	
	@Override
	public int[] prevSize(){
		return new int[]{1, size};
	}
	
	@Override
	public void init(int inputSize, int numTimeSteps){
		size = inputSize;
		
		// initialize weights/biases and their gradient accumulators
		resetW = new Tensor(new int[]{size, size}, true);
		updateW = new Tensor(new int[]{size, size}, true);
		memoryW = new Tensor(new int[]{size, size}, true);
		
		resetU = new Tensor(new int[]{size, size}, true);
		updateU = new Tensor(new int[]{size, size}, true);
		memoryU = new Tensor(new int[]{size, size}, true);
		
		if(useBias){
			resetB = new Tensor(new int[]{1, size}, false);
			updateB = new Tensor(new int[]{1, size}, false);
			memoryB = new Tensor(new int[]{1, size}, false);
		}
		
		gradResetW = new Tensor(new int[]{size, size}, false);
		gradUpdateW = new Tensor(new int[]{size, size}, false);
		gradMemoryW = new Tensor(new int[]{size, size}, false);
		
		gradResetU = new Tensor(new int[]{size, size}, false);
		gradUpdateU = new Tensor(new int[]{size, size}, false);
		gradMemoryU = new Tensor(new int[]{size, size}, false);
		
		if(useBias){
			gradResetB = new Tensor(new int[]{1, size}, false);
			gradUpdateB = new Tensor(new int[]{1, size}, false);
			gradMemoryB = new Tensor(new int[]{1, size}, false);
		}
		
		// used to cache computed results
		reset = new Tensor[numTimeSteps];
		update = new Tensor[numTimeSteps];
		memory = new Tensor[numTimeSteps];
	}
	
	@Override
	public Tensor forwardPropagate(int t, Tensor input, Tensor prevState, boolean training){
		// forward propagate equations
		// omits some details regarding matrix multiplications and stuff
		// reset = sigmoid(input * resetW + prevState * resetU + resetB)
		// update = sigmoid(input * updateW + prevState * updateU + updateB)
		// memory = tanh(input * memoryW + (prevState * reset) * memoryU + memoryB)
		// state = (1 - update) * memory + update * prevState
		
		if(useBias)
			reset[t] = Activation.sigmoid.activate(resetW.dot(input).add(resetU.dot(prevState)).add(resetB));
		else
			reset[t] = Activation.sigmoid.activate(resetW.dot(input).add(resetU.dot(prevState)));
		
		if(useBias)
			update[t] = Activation.sigmoid.activate(updateW.dot(input).add(updateU.dot(prevState)).add(updateB));
		else
			update[t] = Activation.sigmoid.activate(updateW.dot(input).add(updateU.dot(prevState)));
		
		// the activation here can be something other than tanh
		if(useBias)
			memory[t] = activation.activate(memoryW.dot(input).add(memoryU.dot(prevState.mul(reset[t]))).add(memoryB));
		else
			memory[t] = activation.activate(memoryW.dot(input).add(memoryU.dot(prevState.mul(reset[t]))));
		
		return update[t].map(x -> 1.0 - x).mul(memory[t]).add(update[t].mul(prevState));
	}
	
	@Override
	public Tensor[] backPropagate(int t, Tensor input, Tensor prevState, Tensor error){
		// first, gather gradients for the memory, reset, and update equations (multiplied by activation derivatives)
		// second, calculate gradients wrt weights/biases
		// third, accumulate gradients wrt prevState and inputs
		
		Tensor gradMemory = error.mul(update[t].map(x -> 1.0 - x)).mul(activation.derivative(memory[t]));
		
		Tensor gradUpdate = error.mul(prevState.sub(memory[t])).mul(Activation.sigmoid.derivative(update[t]));
		
		Tensor gradReset = memoryU.T().dot(gradMemory).mul(prevState).mul(Activation.sigmoid.derivative(reset[t]));
		
		gradResetW = gradResetW.add(gradReset.dot(input.T()));
		gradResetU = gradResetU.add(gradReset.dot(prevState.T()));
		
		gradUpdateW = gradUpdateW.add(gradUpdate.dot(input.T()));
		gradUpdateU = gradUpdateU.add(gradUpdate.dot(prevState.T()));
		
		gradMemoryW = gradMemoryW.add(gradMemory.dot(input.T()));
		gradMemoryU = gradMemoryU.add(gradMemory.dot(prevState.mul(reset[t]).T()));
		
		if(useBias){
			gradResetB = gradResetB.add(gradReset);
			gradUpdateB = gradUpdateB.add(gradUpdate);
			gradMemoryB = gradMemoryB.add(gradMemory);
		}
		
		Tensor gradInput = resetW.T().dot(gradReset).add(
				updateW.T().dot(gradUpdate)).add(memoryW.T().dot(gradMemory));
		
		Tensor gradPrevState = resetU.T().dot(gradReset).add(
				updateU.T().dot(gradUpdate)).add(memoryU.T().dot(gradMemory).mul(reset[t])).add(error.mul(update[t]));
		
		return new Tensor[]{gradInput, gradPrevState};
	}
	
	@Override
	public void update(Optimizer optimizer, Regularizer regularizer, int changeCount){
		// initialize all extra parameters that are used for optimization
		if(resetWParams == null){
			resetWParams = new Tensor[optimizer.extraParams()];
			updateWParams = new Tensor[optimizer.extraParams()];
			memoryWParams = new Tensor[optimizer.extraParams()];
			
			resetUParams = new Tensor[optimizer.extraParams()];
			updateUParams = new Tensor[optimizer.extraParams()];
			memoryUParams = new Tensor[optimizer.extraParams()];
			
			// use arrays to make initializing tensors take up less lines of code
			Tensor[][] params = {resetWParams, updateWParams, memoryWParams, resetUParams, updateUParams, memoryUParams};
			Tensor[] weights = {resetW, updateW, memoryW, resetU, updateU, memoryU};
			
			for(int i = 0; i < params.length; i++){
				for(int j = 0; j < params[i].length; j++){
					params[i][j] = new Tensor(weights[i].shape(), false);
				}
			}
			
			if(useBias){
				resetBParams = new Tensor[optimizer.extraParams()];
				updateBParams = new Tensor[optimizer.extraParams()];
				memoryBParams = new Tensor[optimizer.extraParams()];
				
				params = new Tensor[][]{resetBParams, updateBParams, memoryBParams};
				weights = new Tensor[]{resetB, updateB, memoryB};
				
				for(int i = 0; i < params.length; i++){
					for(int j = 0; j < params[i].length; j++){
						params[i][j] = new Tensor(weights[i].shape(), false);
					}
				}
			}
		}
		
		// average grads
		gradResetW = gradResetW.div(Math.max(changeCount, 1));
		gradUpdateW = gradUpdateW.div(Math.max(changeCount, 1));
		gradMemoryW = gradMemoryW.div(Math.max(changeCount, 1));
		gradResetU = gradResetU.div(Math.max(changeCount, 1));
		gradUpdateU = gradUpdateU.div(Math.max(changeCount, 1));
		gradMemoryU = gradMemoryU.div(Math.max(changeCount, 1));
		
		// optimize weights using the grads
		if(regularizer == null){
			resetW = resetW.sub(optimizer.optimize(gradResetW, resetWParams));
			updateW = updateW.sub(optimizer.optimize(gradUpdateW, updateWParams));
			memoryW = memoryW.sub(optimizer.optimize(gradMemoryW, memoryWParams));
			
			resetU = resetU.sub(optimizer.optimize(gradResetU, resetUParams));
			updateU = updateU.sub(optimizer.optimize(gradUpdateU, updateUParams));
			memoryU = memoryU.sub(optimizer.optimize(gradMemoryU, memoryUParams));
		}else{
			resetW = resetW.sub(optimizer.optimize(gradResetW.add(regularizer.derivative(resetW)), resetWParams));
			updateW = updateW.sub(optimizer.optimize(gradUpdateW.add(regularizer.derivative(updateW)), updateWParams));
			memoryW = memoryW.sub(optimizer.optimize(gradMemoryW.add(regularizer.derivative(memoryW)), memoryWParams));
			
			resetU = resetU.sub(optimizer.optimize(gradResetU.add(regularizer.derivative(resetU)), resetUParams));
			updateU = updateU.sub(optimizer.optimize(gradUpdateU.add(regularizer.derivative(updateU)), updateUParams));
			memoryU = memoryU.sub(optimizer.optimize(gradMemoryU.add(regularizer.derivative(memoryU)), memoryUParams));
		}
		
		// reset grads
		gradResetW = new Tensor(gradResetW.shape(), false);
		gradUpdateW = new Tensor(gradUpdateW.shape(), false);
		gradMemoryW = new Tensor(gradMemoryW.shape(), false);
		gradResetU = new Tensor(gradResetU.shape(), false);
		gradUpdateU = new Tensor(gradUpdateU.shape(), false);
		gradMemoryU = new Tensor(gradMemoryU.shape(), false);
		
		if(useBias){
			// average grads
			gradResetB = gradResetB.div(Math.max(changeCount, 1));
			gradUpdateB = gradUpdateB.div(Math.max(changeCount, 1));
			gradMemoryB = gradMemoryB.div(Math.max(changeCount, 1));
			
			// optimize biases using the grads
			resetB = resetB.sub(optimizer.optimize(gradResetB, resetBParams));
			updateB = updateB.sub(optimizer.optimize(gradUpdateB, updateBParams));
			memoryB = memoryB.sub(optimizer.optimize(gradMemoryB, memoryBParams));
			
			// reset grads
			gradResetB = new Tensor(gradResetB.shape(), false);
			gradUpdateB = new Tensor(gradUpdateB.shape(), false);
			gradMemoryB = new Tensor(gradMemoryB.shape(), false);
		}
	}
	
	@Override
	public int byteSize(){
		return Double.BYTES * resetW.size() + Double.BYTES * updateW.size() + Double.BYTES * memoryW.size() +
				Double.BYTES * resetU.size() + Double.BYTES * updateU.size() + Double.BYTES * memoryU.size() +
				(useBias ? (Double.BYTES * resetB.size() + Double.BYTES * updateB.size() + Double.BYTES * memoryB.size()) : 0);
	}
	
	@Override
	public ByteBuffer bytes(){
		ByteBuffer bb = ByteBuffer.allocate(byteSize());
		
		Tensor[] weights = {resetW, updateW, memoryW, resetU, updateU, memoryU};
		
		for(Tensor w : weights){
			for(int i = 0; i < w.size(); i++){
				bb.putDouble(w.flatGet(i));
			}
		}
		
		if(useBias){
			weights = new Tensor[]{resetB, updateB, memoryB};
			
			for(Tensor w : weights){
				for(int i = 0; i < w.size(); i++){
					bb.putDouble(w.flatGet(i));
				}
			}
		}
		
		bb.flip();
		return bb;
	}
	
	@Override
	public void readBytes(ByteBuffer bb){
		double[] rW = new double[resetW.size()];
		for(int i = 0; i < rW.length; i++){
			rW[i] = bb.getDouble();
		}
		resetW = new Tensor(resetW.shape(), rW);
		
		double[] uW = new double[updateW.size()];
		for(int i = 0; i < uW.length; i++){
			uW[i] = bb.getDouble();
		}
		updateW = new Tensor(updateW.shape(), uW);
		
		double[] mW = new double[memoryW.size()];
		for(int i = 0; i < mW.length; i++){
			mW[i] = bb.getDouble();
		}
		memoryW = new Tensor(memoryW.shape(), mW);
		
		double[] rU = new double[resetU.size()];
		for(int i = 0; i < rU.length; i++){
			rU[i] = bb.getDouble();
		}
		resetU = new Tensor(resetU.shape(), rU);
		
		double[] uU = new double[updateU.size()];
		for(int i = 0; i < uU.length; i++){
			uU[i] = bb.getDouble();
		}
		updateU = new Tensor(updateU.shape(), uU);
		
		double[] mU = new double[memoryU.size()];
		for(int i = 0; i < mU.length; i++){
			mU[i] = bb.getDouble();
		}
		memoryU = new Tensor(memoryU.shape(), uU);
		
		if(useBias){
			double[] rB = new double[resetB.size()];
			for(int i = 0; i < rB.length; i++){
				rB[i] = bb.getDouble();
			}
			resetB = new Tensor(resetB.shape(), rB);
			
			double[] uB = new double[updateB.size()];
			for(int i = 0; i < uB.length; i++){
				uB[i] = bb.getDouble();
			}
			updateB = new Tensor(updateB.shape(), uB);
			
			double[] mB = new double[memoryB.size()];
			for(int i = 0; i < mB.length; i++){
				mB[i] = bb.getDouble();
			}
			memoryB = new Tensor(memoryB.shape(), mB);
		}
	}
}
