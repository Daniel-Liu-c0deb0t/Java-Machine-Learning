package utils;

public interface Loss{
	public static final Loss squared = new Loss(){
		@Override
		public double loss(double[] x, double[] t){
			double result = 0.0;
			for(int i = 0; i < t.length; i++){
				result += Math.pow(t[i] - x[i], 2.0);
			}
			return result;
		}
		
		@Override
		public double[] derivative(double[] x, double[] t){
			double[] result = new double[t.length];
			for(int i = 0; i < t.length; i++){
				result[i] = x[i] - t[i];
			}
			return result;
		}
	};
	
	public static final Loss crossEntropy = new Loss(){
		@Override
		public double loss(double[] x, double[] t){
			return -Math.log(x[UtilMethods.argMax(t)]);
		}
		
		@Override
		public double[] derivative(double[] x, double[] t){
			double[] result = new double[t.length];
			for(int i = 0; i < t.length; i++){
				result[i] = x[i] - t[i];
			}
			return result;
		}
	};
	
	public double loss(double[] x, double[] t);
	public double[] derivative(double[] x, double[] t);
}
