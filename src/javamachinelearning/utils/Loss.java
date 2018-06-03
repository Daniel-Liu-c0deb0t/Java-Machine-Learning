package javamachinelearning.utils;

public interface Loss{
	public static final Loss squared = new Loss(){
		@Override
		public Tensor loss(Tensor x, Tensor t){
			return x.sub(t).map(val -> val * val);
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			return x.sub(t);
		}
		
		@Override
		public String toString(){
			return "Squared";
		}
	};
	
	// multi-class classification
	public static final Loss softmaxCrossEntropy = new Loss(){
		@Override
		public Tensor loss(Tensor x, Tensor t){
			// because the target is a one hot vector
			return t.mul(x.map(val -> -Math.log(val)));
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			// because the output layer has to be softmax
			return x.sub(t);
		}
		
		@Override
		public String toString(){
			return "Softmax Cross Entropy";
		}
	};
	
	// binary classification
	public static final Loss binaryCrossEntropy = new Loss(){
		@Override
		public Tensor loss(Tensor x, Tensor t){
			// because the target is 0 or 1
			Tensor a = t.mul(x.map(val -> -Math.log(val)));
			Tensor b = t.map(val -> 1.0 - val).mul(x.map(val -> -Math.log(1.0 - val)));
			return a.add(b);
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			// if output layer is sigmoid, the denominator cancels out
			return x.sub(t).div(x.map(val -> val * (1.0 - val)));
		}
		
		@Override
		public String toString(){
			return "Binary Cross Entropy";
		}
	};
	
	public Tensor loss(Tensor x, Tensor t);
	public Tensor derivative(Tensor x, Tensor t);
}
