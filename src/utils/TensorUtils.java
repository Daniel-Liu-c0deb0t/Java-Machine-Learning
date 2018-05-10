package utils;

public class TensorUtils{
	// simple way of creating 1D tensors
	public static Tensor t(double... vals){
		return new Tensor(vals);
	}
}
