package utils;

public interface Loss{
	public static final Loss squared = new Loss(){
		@Override
		public double loss(Tensor x, Tensor t){
			return x.sub(t).reduce(0, (a, b) -> a + b * b);
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			return x.sub(t);
		}
	};
	
	// multi-class classification
	public static final Loss softmaxCrossEntropy = new Loss(){
		@Override
		public double loss(Tensor x, Tensor t){
			// because the target is a one hot vector
			return -Math.log(x.flatGet(Utils.argMax(t)));
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			// because the output layer has to be softmax
			return x.sub(t);
		}
	};
	
	// binary classification
	public static final Loss binaryCrossEntropy = new Loss(){
		@Override
		public double loss(Tensor x, Tensor t){
			// because the target is 0 or 1
			if(t.flatGet(0) == 0.0)
				return Math.log(1.0 - x.flatGet(0));
			else
				return Math.log(x.flatGet(0));
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			// if output layer is sigmoid, the denominator cancels out
			return x.sub(t).div(x.map(val -> val * (1.0 - val)));
		}
	};
	
	public double loss(Tensor x, Tensor t);
	public Tensor derivative(Tensor x, Tensor t);
}
