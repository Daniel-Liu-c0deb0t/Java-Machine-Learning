package javamachinelearning.utils;

import static javamachinelearning.utils.TensorUtils.*;

import java.util.Arrays;
import java.util.Random;

public class Utils{
	public static String format(double num){
		return String.format("%,.7g", num);
	}
	
	public static String shorterFormat(double num){
		return String.format("%,.2g", num);
	}
	
	public static String formatElapsedTime(long ms){
		return String.format("%02d:%02d:%02d.%03d", ms / (3600 * 1000), ms / (60 * 1000) % 60, ms / 1000 % 60, ms % 1000);
	}
	
	public static void printArray(double[][] arr){
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++){
				System.out.print(format(arr[i][j]) + " ");
			}
			System.out.println();
		}
	}
	
	public static void printArray(double[] arr){
		for(int i = 0; i < arr.length; i++){
			System.out.print(format(arr[i]) + " ");
		}
		System.out.println();
	}
	
	public static void printArray(double[] arr, int count){
		for(int i = 0; i < count; i++){
			System.out.print(format(arr[i]) + " ");
		}
		System.out.println();
	}
	
	public static String makeStr(char c, int n){
		char[] result = new char[n];
		Arrays.fill(result, c);
		return new String(result);
	}
	
	public static int unsignedByteToInt(byte b){
		int result = 0;
		for(int i = 0; i < 8; i++){
			if((b & (1 << i)) != 0)
				result += 1 << i;
		}
		return result;
	}
	
	public static void printImage(Tensor[] image){
		char[][][] chars = new char[image.length][0][0];
		for(int i = 0; i < image.length; i++){
			int s1 = image[i].shape()[0];
			int s2 = image[i].shape()[1];
			chars[i] = new char[s2][s1];
			for(int j = 0; j < s1; j++){
				for(int k = 0; k < s2; k++){
					if(image[i].flatGet(j * s2 + k) < 0.3){
						chars[i][k][j] = ' ';
					}else if(image[i].flatGet(j * s2 + k) > 0.6){
						chars[i][k][j] = '#';
					}else{
						chars[i][k][j] = '.';
					}
				}
			}
		}
		for(int i = 0; i < chars.length; i++){
			for(int j = 0; j < chars[i].length; j++){
				for(int k = 0; k < chars[i][j].length; k++){
					System.out.print(chars[i][j][k] + " ");
				}
				System.out.println();
			}
			System.out.println();
		}
	}
	
	public static double classificationAccuracy(Tensor[] output, Tensor[] target){
		int totalCorrect = 0;
		for(int i = 0; i < output.length; i++){
			if(argMax(output[i]) == argMax(target[i])){
				totalCorrect++;
			}
		}
		return (double)totalCorrect / (double)output.length;
	}
	
	public static double averageDeviation(Tensor[] output, Tensor[] target){
		double sum = 0.0;
		for(int i = 0; i < output.length; i++){
			sum += Math.sqrt(Loss.squared.loss(output[i], target[i]));
		}
		return (double)sum / (double)output.length;
	}
	
	public static int[] centerOfMass(double[][] arr){
		double sumX = 0.0;
		double sumY = 0.0;
		double total = 0.0;
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++){
				if(arr[i][j] > 0.0){
					sumX += i * arr[i][j];
					sumY += j * arr[i][j];
					total += arr[i][j];
				}
			}
		}
		return new int[]{(int)(sumY / total), (int)(sumX / total)};
	}
	
	public static double[] flatCombine(Tensor... tensors){
		int sum = 0;
		for(int i = 0; i < tensors.length; i++){
			sum += tensors[i].size();
		}
		
		double[] res = new double[sum];
		int idx = 0;
		for(int i = 0; i < tensors.length; i++){
			for(int j = 0; j < tensors[i].size(); j++){
				res[idx] = tensors[i].flatGet(j);
				idx++;
			}
		}
		return res;
	}
	
	public static Tensor[] flattenAll(Tensor[] tensors){
		Tensor[] res = new Tensor[tensors.length];
		for(int i = 0; i < tensors.length; i++){
			res[i] = tensors[i].flatten();
		}
		return res;
	}
	
	public static Tensor[] reshapeAll(Tensor[] tensors, int... shape){
		Tensor[] res = new Tensor[tensors.length];
		for(int i = 0; i < tensors.length; i++){
			res[i] = tensors[i].reshape(shape);
		}
		return res;
	}
	
	public static Tensor centerData(double[][] arr, int width, int height){
		double[][] centeredArr = new double[height][width];
		int[] centerOfMass = centerOfMass(arr);
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++){
				if(arr[i][j] > 0.0){
					int y = (height - arr.length) / 2 + i + arr.length / 2 - centerOfMass[0];
					int x = (width - arr[i].length) / 2 + j + arr[i].length / 2 - centerOfMass[1];
					if(y >= 0 && y < height && x >= 0 && x < width)
						centeredArr[y][x] = arr[i][j];
				}
			}
		}
		return new Tensor(centeredArr);
	}
	
	public static Tensor[] standardDist(double x, double y, double s, int n){
		Tensor[] res = new Tensor[n];
		Random r = new Random();
		for(int i = 0; i < n; i++){
			res[i] = t(x + r.nextGaussian() * s, y + r.nextGaussian() * s);
		}
		return res;
	}
	
	public static Tensor[] concat(Tensor[]... tensors){
		int sum = 0;
		for(int i = 0; i < tensors.length; i++){
			sum += tensors[i].length;
		}
		Tensor[] res = new Tensor[sum];
		int idx = 0;
		for(int i = 0; i < tensors.length; i++){
			for(int j = 0; j < tensors[i].length; j++){
				res[idx] = tensors[i][j];
				idx++;
			}
		}
		return res;
	}
	
	public static void shuffle(Tensor[] x, Tensor[] y){
		Random r = new Random();
		for(int i = x.length - 1; i > 0; i--){
			int j = r.nextInt(i + 1);
			Tensor xTemp = x[i];
			Tensor yTemp = y[i];
			x[i] = x[j];
			y[i] = y[j];
			x[j] = xTemp;
			y[j] = yTemp;
		}
	}
}
