package utils;

public interface Activation{
	public static final Activation linear = new Activation(){
		@Override
		public double activate(double x, double[] arr){
			return x;
		}
		
		@Override
		public double derivative(double y){
			return 1.0;
		}
	};
	
	public static final Activation sigmoid = new Activation(){
		@Override
		public double activate(double x, double[] arr){
			return 1.0 / (1.0 + Math.exp(-x));
		}
		
		@Override
		public double derivative(double y){
			return y * (1.0 - y);
		}
	};
	
	public static final Activation tanh = new Activation(){
		@Override
		public double activate(double x, double[] arr){
			return (1.0 - Math.exp(x * -2.0)) / (1.0 + Math.exp(x * 2.0));
		}
		
		@Override
		public double derivative(double y){
			return 1.0 - Math.pow(y, 2.0);
		}
	};
	
	public static final Activation relu = new Activation(){
		@Override
		public double activate(double x, double[] arr){
			return Math.max(0.0, x);
		}
		
		@Override
		public double derivative(double y){
			return y > 0.0 ? 1.0 : 0.0;
		}
	};
	
	public static final Activation exp = new Activation(){
		@Override
		public double activate(double x, double[] arr){
			return Math.exp(x);
		}
		
		@Override
		public double derivative(double y){
			return y;
		}
	};
	
	public static final Activation softmax = new Activation(){
		@Override
		public double activate(double x, double[] arr){
			double sum = 0.0;
			double offset = 0.0;
			for(int i = 0; i < arr.length; i++){
				offset = Math.max(offset, arr[i]);
			}
			for(int i = 0; i < arr.length; i++){
				sum += Math.exp(arr[i] - offset);
			}
			return Math.exp(x - offset) / sum;
		}
		
		@Override
		public double derivative(double y){
			return 1.0;
		}
	};
	
	public double activate(double val, double[] arr);
	public double derivative(double val);
}
