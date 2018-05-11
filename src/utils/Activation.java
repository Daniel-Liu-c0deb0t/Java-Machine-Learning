package utils;

public interface Activation{
	public static final Activation linear = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t;
		}
		
		@Override
		public Tensor derivative(Tensor t){
			return new Tensor(t.shape(), 1.0);
		}
	};
	
	public static final Activation sigmoid = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> 1.0 / (1.0 + Math.exp(-x)));
		}
		
		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> x * (1.0 - x));
		}
	};
	
	public static final Activation tanh = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> 2.0 / (1.0 + Math.exp(-2.0 * x)) - 1.0);
		}
		
		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> 1.0 - x * x);
		}
	};
	
	public static final Activation relu = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> Math.max(0.0, x));
		}
		
		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> x > 0.0 ? 1.0 : 0.0);
		}
	};
	
	public static final Activation softmax = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			double max = t.reduce(Double.MIN_VALUE, (a, b) -> Math.max(a, b));
			double sum = t.reduce(0, (a, b) -> a + Math.exp(b - max));
			return t.map(x -> Math.exp(x - max) / sum);
		}
		
		@Override
		public Tensor derivative(Tensor t){
			return new Tensor(t.shape(), 1.0);
		}
	};
	
	public Tensor activate(Tensor t);
	public Tensor derivative(Tensor t);
}
