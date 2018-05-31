package javamachinelearning.utils;

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
			Tensor max = t.reduceLast(Double.MIN_VALUE, (a, b) -> Math.max(a, b));
			max = max.dupLast(t.shape()[t.shape().length - 1]);
			
			Tensor exp = t.sub(max).map(x -> Math.exp(x));
			
			Tensor sum = exp.reduceLast(0, (a, b) -> a + b);
			sum = sum.dupLast(t.shape()[t.shape().length - 1]);
			
			return exp.div(sum);
		}
		
		@Override
		public Tensor derivative(Tensor t){
			return new Tensor(t.shape(), 1.0);
		}
	};
	
	public Tensor activate(Tensor t);
	public Tensor derivative(Tensor t);
}
