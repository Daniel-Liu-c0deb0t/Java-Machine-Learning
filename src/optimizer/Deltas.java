package optimizer;

public class Deltas{
	private double[][] delta1;
	private double[][] delta2;
	
	public Deltas(double[][] delta1, double[][] delta2){
		this.delta1 = delta1;
		this.delta2 = delta2;
	}
	
	public double[][] getDelta1(){
		return delta1;
	}
	
	public double[][] getDelta2(){
		return delta2;
	}
}
