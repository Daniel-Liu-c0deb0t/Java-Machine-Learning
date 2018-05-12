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
	
	public static final Loss crossEntropy = new Loss(){
		@Override
		public double loss(Tensor x, Tensor t){
			//because the target is a one hot vector
			return -Math.log(x.flatGet(UtilMethods.argMax(t)));
		}
		
		@Override
		public Tensor derivative(Tensor x, Tensor t){
			//because the output layer should be softmax
			return x.sub(t);
		}
	};
	
	public double loss(Tensor x, Tensor t);
	public Tensor derivative(Tensor x, Tensor t);
}
