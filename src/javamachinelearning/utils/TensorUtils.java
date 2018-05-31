package javamachinelearning.utils;

public class TensorUtils{
	// simple way of creating column vectors
	public static Tensor t(double... vals){
		return new Tensor(new int[]{1, vals.length}, vals);
	}
	
	public static Tensor oneHot(int idx, int size){
		double[] res = new double[size];
		res[idx] = 1.0;
		return new Tensor(res);
	}
	
	public static Tensor oneHotString(String s, String alphabet){
		Tensor[] res = new Tensor[s.length()];
		for(int i = 0; i < s.length(); i++){
			res[i] = oneHot(alphabet.indexOf(s.charAt(i)), alphabet.length());
		}
		return stack(res);
	}
	
	// decode a one hot string
	public static String decodeString(Tensor val, boolean rand, String alphabet){
		char[] res = new char[val.shape()[0]];
		for(int i = 0; i < val.shape()[0]; i++){
			res[i] = alphabet.charAt(rand ? randProb(val.get(i)) : argMax(val.get(i)));
		}
		return new String(res);
	}
	
	// pick random with probabilities
	public static int randProb(Tensor tensor){
		double[] pre = new double[tensor.size()];
		for(int i = 0; i < pre.length; i++){
			pre[i] = tensor.flatGet(i) + (i == 0 ? 0 : pre[i - 1]);
		}
		
		double rand = Math.random() * pre[pre.length - 1];
		for(int i = 0; i < pre.length; i++){
			if(pre[i] >= rand)
				return i;
		}
		return -1;
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
		int[] shape;
		if(tensors[0].shape()[0] == 1 && tensors[0].shape().length == 2){
			shape = new int[]{tensors.length, tensors[0].shape()[1]};
		}else{
			shape = new int[tensors[0].shape().length + 1];
			shape[0] = tensors.length;
			
			for(int i = 0; i < tensors[0].shape().length; i++){
				shape[i + 1] = tensors[0].shape()[i];
			}
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
