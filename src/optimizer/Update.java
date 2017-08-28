package optimizer;

public class Update{
	private double[] error;
	private double[] error2;
	
	public Update(double[] error, double[] error2){
		this.error = error;
		this.error2 = error2;
	}
	
	public double[] getError(){
		return error;
	}
	
	public double[] getError2(){
		return error2;
	}
}
