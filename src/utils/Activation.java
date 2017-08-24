package utils;

public interface Activation{
	public static final Activation linear = (x, arr) -> {
		return x;
	};
	
	public static final Activation sigmoid = (x, arr) -> {
		return 1.0 / (1.0 + Math.exp(-x));
	};
	
	public static final Activation tanh = (x, arr) -> {
		return (1.0 - Math.exp(x * -2.0)) / (1.0 + Math.exp(x * 2.0));
	};
	
	public static final Activation relu = (x, arr) -> {
		return Math.max(0.0, x);
	};
	
	public static final Activation exp = (x, arr) -> {
		return Math.exp(x);
	};
	
	public static final Activation softmax = (x, arr) -> {
		double sum = 0.0;
		for(int i = 0; i < arr.length; i++){
			sum += Math.exp(arr[i]);
		}
		return Math.exp(x) / sum;
	};
	
	public static final Activation linearP = (y, arr) -> {
		return 1.0;
	};
	
	public static final Activation sigmoidP = (y, arr) -> {
		return y * (1.0 - y);
	};
	
	public static final Activation tanhP = (y, arr) -> {
		return 1.0 - Math.pow(y, 2.0);
	};
	
	public static final Activation reluP = (y, arr) -> {
		return y > 0.0 ? 1.0 : 0.0;
	};
	
	public static final Activation expP = (y, arr) -> {
		return y;
	};
	
	public static final Activation softmaxP = (y, arr) -> {
		return y - arr[0];
	};
	
	public double activate(double val, double[] arr);
}
