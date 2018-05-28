package javamachinelearning.layers;

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
	private Tensor reset, update, memory;
	
	private boolean useBias = true;
	private int changeCount;
	
	public GRUCell(int size, Activation activation){
		this.size = size;
		this.activation = activation;
		
		resetW = new Tensor(new int[]{size, size}, true);
		updateW = new Tensor(new int[]{size, size}, true);
		memoryW = new Tensor(new int[]{size, size}, true);
		
		resetU = new Tensor(new int[]{size, size}, true);
		updateU = new Tensor(new int[]{size, size}, true);
		memoryU = new Tensor(new int[]{size, size}, true);
		
		resetB = new Tensor(new int[]{size}, false);
		updateB = new Tensor(new int[]{size}, false);
		memoryB = new Tensor(new int[]{size}, false);
		
		gradResetW = new Tensor(new int[]{size, size}, false);
		gradUpdateW = new Tensor(new int[]{size, size}, false);
		gradMemoryW = new Tensor(new int[]{size, size}, false);
		
		gradResetU = new Tensor(new int[]{size, size}, false);
		gradUpdateU = new Tensor(new int[]{size, size}, false);
		gradMemoryU = new Tensor(new int[]{size, size}, false);
		
		gradResetB = new Tensor(new int[]{size}, false);
		gradUpdateB = new Tensor(new int[]{size}, false);
		gradMemoryB = new Tensor(new int[]{size}, false);
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
		return new int[]{size};
	}
	
	@Override
	public int[] prevSize(){
		return new int[]{size};
	}
	
	@Override
	public Tensor forwardPropagate(Tensor input, Tensor prevState, boolean training){
		// forward propagate equations
		// omits some details regarding matrix multiplications and stuff
		// reset = sigmoid(input * resetW + prevState * resetU + resetB)
		// update = sigmoid(input * updateW + prevState * updateU + updateB)
		// memory = tanh(input * memoryW + (prevState * reset) * memoryU + memoryB)
		// state = (1 - update) * memory + update * prevState
		
		if(useBias)
			reset = Activation.sigmoid.activate(resetW.dot(input).add(resetU.dot(prevState)).add(resetB));
		else
			reset = Activation.sigmoid.activate(resetW.dot(input).add(resetU.dot(prevState)));
		
		if(useBias)
			update = Activation.sigmoid.activate(updateW.dot(input).add(updateU.dot(prevState)).add(updateB));
		else
			update = Activation.sigmoid.activate(updateW.dot(input).add(updateU.dot(prevState)));
		
		// the activation here can be something other than tanh
		if(useBias)
			memory = activation.activate(memoryW.dot(input).add(memoryU.dot(prevState.mul(reset))).add(memoryB));
		else
			memory = activation.activate(memoryW.dot(input).add(memoryU.dot(prevState.mul(reset))));
		
		return update.map(x -> 1.0 - x).mul(memory).add(update.mul(prevState));
	}
	
	@Override
	public Tensor[] backPropagate(Tensor input, Tensor prevState, Tensor error){
		// first, gather gradients for the memory, reset, and update equations (multiplied by activation derivatives)
		// second, calculate gradients wrt weights/biases
		// third, accumulate gradients wrt prevState and inputs
		
		Tensor gradMemory = error.mul(update.map(x -> 1.0 - x)).mul(activation.derivative(memory));
		
		Tensor gradUpdate = error.mul(prevState.sub(memory)).mul(Activation.sigmoid.derivative(update));
		
		Tensor gradReset = memoryU.T().dot(gradMemory).mul(prevState).mul(Activation.sigmoid.derivative(reset));
		
		gradResetW = gradResetW.add(input.mulEach(gradReset));
		gradResetU = gradResetU.add(prevState.mulEach(gradReset));
		
		gradUpdateW = gradUpdateW.add(input.mulEach(gradUpdate));
		gradUpdateU = gradUpdateU.add(prevState.mulEach(gradUpdate));
		
		gradMemoryW = gradMemoryW.add(input.mulEach(gradMemory));
		gradMemoryU = gradMemoryU.add(prevState.mul(reset).mulEach(gradMemory));
		
		if(useBias){
			gradResetB = gradResetB.add(gradReset);
			gradUpdateB = gradUpdateB.add(gradUpdate);
			gradMemoryB = gradMemoryB.add(gradMemory);
		}
		
		Tensor gradInput = 
		
		changeCount++;
	}
	
	@Override
	public void update(Optimizer optimizer, Regularizer regularizer){
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
		
		changeCount = 0;
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
