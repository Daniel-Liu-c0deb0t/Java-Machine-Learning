package utils;

public interface Loss{
	public static final Loss squared = (x, t, reg) -> {
		double[] result = new double[t.length];
		for(int i = 0; i < t.length; i++){
			result[i] = x[i] - t[i] + reg;
		}
		return result;
	};
	
	public static final Loss crossEntropy = (x, t, reg) -> {
		double[] result = new double[t.length];
		for(int i = 0; i < t.length; i++){
			result[i] = x[i] - t[i] + reg;
		}
		return result;
	};
	
	public double[] loss(double[] x, double[] t, double reg);
}
