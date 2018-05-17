package javamachinelearning.utils;

import java.util.Random;

public class Tensor{
	private int[] shape;
	private double[] data;
	
	private int[] mult;
	private int size;
	
	public Tensor(int[] shape, boolean rand){
		this.shape = shape;
		
		calcMult();
		
		data = new double[size];
		if(rand){
			Random r = new Random();
			// for initializing weights
			int sum = 0;
			for(int i = 0; i < shape.length; i++){
				sum += shape[i];
			}
			for(int i = 0; i < size; i++){
				// xavier normal initialization (not truncated)
				data[i] = r.nextGaussian() * Math.sqrt(2.0 / sum);
			}
		}else{
			for(int i = 0; i < size; i++){
				data[i] = 0;
			}
		}
	}
	
	public Tensor(int[] shape, double init){
		this.shape = shape;
		
		calcMult();
		
		data = new double[size];
		for(int i = 0; i < size; i++){
			data[i] = init;
		}
	}
	
	public Tensor(double[] d){
		shape = new int[]{d.length};
		calcMult();
		data = new double[size];
		for(int i = 0; i < d.length; i++){
			data[i] = d[i];
		}
	}
	
	// note that the following two initializers work in row major format!
	// however, the data is internally represented as column major, so some swaps happen
	public Tensor(double[][] d){
		shape = new int[]{d[0].length, d.length};
		calcMult();
		data = new double[size];
		int idx = 0;
		for(int i = 0; i < d[0].length; i++){
			for(int j = 0; j < d.length; j++){
				data[idx] = d[j][i];
				idx++;
			}
		}
	}
	
	// the first dimension is treated as the depth!
	public Tensor(double[][][] d){
		shape = new int[]{d[0][0].length, d[0].length, d.length};
		calcMult();
		data = new double[size];
		int idx = 0;
		for(int i = 0; i < d[0][0].length; i++){
			for(int j = 0; j < d[0].length; j++){
				for(int k = 0; k < d.length; k++){
					data[idx] = d[k][j][i];
					idx++;
				}
			}
		}
	}
	
	public Tensor(int[] shape, double[] data){
		this.shape = shape;
		calcMult();
		this.data = data;
	}
	
	public int[] shape(){
		return shape;
	}
	
	public int[] mult(){
		return mult;
	}
	
	public int size(){
		return size;
	}
	
	public Tensor add(Tensor o){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] + o.data[i];
		}
		return new Tensor(shape, res);
	}
	
	public Tensor add(double d){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] + d;
		}
		return new Tensor(shape, res);
	}
	
	public Tensor sub(Tensor o){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] - o.data[i];
		}
		return new Tensor(shape, res);
	}
	
	public Tensor sub(double d){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] - d;
		}
		return new Tensor(shape, res);
	}
	
	public Tensor mul(Tensor o){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] * o.data[i];
		}
		return new Tensor(shape, res);
	}
	
	public Tensor mul(double d){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] * d;
		}
		return new Tensor(shape, res);
	}
	
	public Tensor div(Tensor o){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] / o.data[i];
		}
		return new Tensor(shape, res);
	}
	
	public Tensor div(double d){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = data[i] / d;
		}
		return new Tensor(shape, res);
	}
	
	public Tensor dot(Tensor o){
		// basically matrix multiply
		// both must be 2D matrices
		// second matrix can be 1D, but it will be treated as 2D
		int[] s2;
		if(o.shape.length == 1)
			s2 = new int[]{1, o.shape[0]};
		else
			s2 = o.shape;
		
		double[] res = new double[s2[0] * shape[1]];
		int idx = 0;
		
		for(int i = 0; i < shape[1]; i++){
			for(int j = 0; j < s2[0]; j++){
				for(int k = 0; k < shape[0]; k++){
					res[idx] += data[k * shape[1] + i] * o.data[j * s2[1] + k];
				}
				idx++;
			}
		}
		
		return new Tensor(new int[]{s2[0], shape[1]}, res);
	}
	
	public Tensor mulEach(Tensor o){ // cartesian product, then multiply the pairs
		double[] res = new double[size * o.size];
		int idx = 0;
		for(int i = 0; i < size; i++){
			for(int j = 0; j < o.size; j++){
				res[idx] = data[i] * o.data[j];
				idx++;
			}
		}
		return new Tensor(new int[]{size, o.size}, res);
	}
	
	public Tensor T(){ // transposes 2D matrix
		double[] res = new double[size];
		int idx = 0;
		for(int i = 0; i < shape[1]; i++){
			for(int j = 0; j < shape[0]; j++){
				res[idx] = data[j * shape[1] + i];
				idx++;
			}
		}
		return new Tensor(new int[]{shape[1], shape[0]}, res);
	}
	
	public Tensor flatten(){
		return new Tensor(new int[]{size}, data);
	}
	
	public Tensor reshape(int... s){
		return new Tensor(s, data);
	}
	
	public Tensor map(Function f){
		double[] res = new double[size];
		for(int i = 0; i < size; i++){
			res[i] = f.apply(data[i]);
		}
		return new Tensor(shape, res);
	}
	
	public double reduce(double init, Function2 f){
		double res = init;
		for(int i = 0; i < size; i++){
			res = f.apply(res, data[i]);
		}
		return res;
	}
	
	public double flatGet(int idx){
		return data[idx];
	}
	
	public interface Function{
		public double apply(double x);
	}
	
	public interface Function2{
		public double apply(double a, double b);
	}
	
	private void calcMult(){
		mult = new int[shape.length];
		mult[shape.length - 1] = 1;
		size = shape[shape.length - 1];
		for(int i = shape.length - 2; i >= 0; i--){
			mult[i] = mult[i + 1] * shape[i + 1];
			size *= shape[i];
		}
	}
	
	@Override
	public Tensor clone(){
		return new Tensor(shape, data);
	}
	
	// toString returns a string that is in column major format!
	@Override
	public String toString(){
		return str(0, size, 0);
	}
	
	private String str(int start, int end, int depth){
		if(depth >= shape.length - 1){
			StringBuilder b = new StringBuilder();
			b.append('[');
			for(int i = start; i < end; i += mult[depth]){
				b.append(Utils.format(data[i]) + ", ");
			}
			if(b.length() > 1)
				b.delete(b.length() - 2, b.length());
			b.append(']');
			return b.toString();
		}
		
		StringBuilder b = new StringBuilder();
		b.append('[');
		for(int i = start; i < end; i += mult[depth]){
			b.append(str(i, i + mult[depth], depth + 1) + ",\n");
			if(depth < shape.length - 2)
				b.append('\n');
		}
		if(b.length() > 1)
			b.delete(b.length() - 2 - (depth < shape.length - 2 ? 1 : 0), b.length());
		b.append(']');
		return b.toString();
	}
}
