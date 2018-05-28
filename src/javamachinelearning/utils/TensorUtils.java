package javamachinelearning.utils;

public class TensorUtils{
	// simple way of creating 1D tensors
	public static Tensor t(double... vals){
		return new Tensor(vals);
	}
	
	public static Tensor oneHotEncode(int idx, int size){
		double[] res = new double[size];
		res[idx] = 1.0;
		return new Tensor(res);
	}
	
	public static int argMax(Tensor tensor){
		double max = Double.MIN_VALUE;
		int maxIndex = -1;
		for(int i = 0; i < tensor.size(); i++){
			if(tensor.flatGet(i) > max){
				max = tensor.flatGet(i);
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static Tensor stack(Tensor... tensors){
		int[] shape = new int[tensors[0].shape().length + 1];
		shape[0] = tensors.length;
		
		for(int i = 0; i < tensors[0].shape().length; i++){
			shape[i + 1] = tensors[0].shape()[i];
		}
		
		double[] res = new double[tensors[0].size() * tensors.length];
		int idx = 0;
		for(int i = 0; i < tensors.length; i++){
			for(int j = 0; j < tensors[i].size(); j++){
				res[idx] = tensors[i].flatGet(j);
				idx++;
			}
		}
		return new Tensor(shape, res);
	}
}
