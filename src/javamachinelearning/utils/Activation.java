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

		@Override
		public String toString(){
			return "Linear";
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

		@Override
		public String toString(){
			return "Sigmoid";
		}
	};
	
	// linear approximation of sigmoid
	public static final Activation hardSigmoid = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> Math.min(Math.max(x * 0.2 + 0.5, 0.0), 1.0));
		}

		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> x > 0.0 && x < 1.0 ? 0.2 : 0.0);
		}

		@Override
		public String toString(){
			return "Hard Sigmoid";
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

		@Override
		public String toString(){
			return "Hyperbolic Tangent";
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

		@Override
		public String toString(){
			return "Rectified Linear Unit";
		}
	};
	
	public static final Activation leakyRelu = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			// note: hard coded leaky value!
			return t.map(x -> x > 0.0 ? x : x * 0.01);
		}

		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> x > 0.0 ? 1.0 : 0.01);
		}

		@Override
		public String toString(){
			return "Leaky Rectified Linear Unit";
		}
	};
	
	public static final Activation relu6 = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> Math.min(Math.max(0.0, x), 6.0));
		}

		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> (x > 0.0) && (x < 6.0) ? 1.0 : 0.0);
		}

		@Override
		public String toString(){
			return "Rectified Linear Unit 6";
		}
	};

	public static final Activation relu3 = new Activation(){
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> Math.min(Math.max(0.0, x), 3.0));
		}

		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> (x > 0.0) && (x < 3.0) ? 1.0 : 0.0);
		}

		@Override
		public String toString(){
			return "Rectified Linear Unit 3";
		}
	};

	public static final Activation elu = new Activation(){
		double alpha = 1.0;
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> Math.max(alpha * (Math.exp(x)-1.0), x));
		}

		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> x > 0 ? 1.0 : alpha * Math.exp(x));
		}

		@Override
		public String toString(){
			return "Exponential Linear Unit";
		}
	};

	public static final Activation selu = new Activation(){
		double alpha = 1.6732632423543772848170429916717;
		double scale = 1.0507009873554804934193349852946;
		@Override
		public Tensor activate(Tensor t){
			return t.map(x -> Math.max(scale * alpha * (Math.exp(x)-1.0), x));
		}

		@Override
		public Tensor derivative(Tensor t){
			return t.map(x -> x > 0 ? 1.0 : scale * alpha * Math.exp(x));
		}

		@Override
		public String toString(){
			return "Scaled Exponential Linear Unit";
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
			// because the loss function should be cross entropy
			return new Tensor(t.shape(), 1.0);
		}

		@Override
		public String toString(){
			return "Softmax";
		}
	};
	
	public Tensor activate(Tensor t);
	// derivatives are calculated in terms of the activated output
	public Tensor derivative(Tensor t);
}
