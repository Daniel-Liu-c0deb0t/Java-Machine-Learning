package utils;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.Random;

import edge.Edge;
import layer.Layer;
import network.NeuralNetwork;

public class UtilMethods{
	public static String format(double num){
		return String.format("%.7g", num);
	}
	
	public static String shorterFormat(double num){
		return String.format("%.3g", num);
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
	
	public static Tensor oneHotEncode(int idx, int size){
		double[] res = new double[size];
		res[idx] = 1.0;
		return new Tensor(res);
	}
	
	public static int argMax(double[] arr){
		double max = 0.0;
		int maxIndex = -1;
		for(int i = 0; i < arr.length; i++){
			if(arr[i] > max){
				max = arr[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int argMax(Tensor t){
		double max = 0.0;
		int maxIndex = -1;
		for(int i = 0; i < t.size(); i++){
			if(t.flatGet(i) > max){
				max = t.flatGet(i);
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public static int unsignedByteToInt(byte b){
		int result = 0;
		for(int i = 0; i < 8; i++){
			if((b & (1 << i)) != 0)
				result += 1 << i;
		}
		return result;
	}
	
	public static void printImage(double[][] image){
		int sideLength = (int)Math.sqrt(image[0].length);
		char[][][] chars = new char[image.length][sideLength][sideLength];
		for(int i = 0; i < image.length; i++){
			for(int j = 0; j < image[i].length; j++){
				if(image[i][j] < 0.3){
					chars[i][j / sideLength][j % sideLength] = ' ';
				}else if(image[i][j] > 0.6){
					chars[i][j / sideLength][j % sideLength] = '#';
				}else{
					chars[i][j / sideLength][j % sideLength] = '.';
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
		return new int[]{(int)(sumX / total), (int)(sumY / total)};
	}
	
	public static double[] flattenData(double[][] arr){
		double[] result = new double[arr.length * arr[0].length];
		for(int i = 0; i < arr[0].length; i++){
			for(int j = 0; j < arr.length; j++){
				result[i * arr.length + j] = arr[j][i];
			}
		}
		return result;
	}
	
	public static double[][] centerData(double[][] arr, int width, int height){
		double[][] centeredArr = new double[width][height];
		int[] centerOfMass = UtilMethods.centerOfMass(arr);
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++){
				if(arr[i][j] > 0.0)
					centeredArr[(width - arr.length) / 2 + i + arr.length / 2 - centerOfMass[0]][(height - arr[i].length) / 2 + j + arr[i].length / 2 - centerOfMass[1]] = arr[i][j];
			}
		}
		return centeredArr;
	}
	
	public static double[][] standardDist(double x, double y, double s, int n){
		double[][] result = new double[n][2];
		Random r = new Random();
		for(int i = 0; i < n; i++){
			result[i][0] = x + r.nextGaussian() * s;
			result[i][1] = y + r.nextGaussian() * s;
		}
		return result;
	}
	
	public static double[][] concat(double[][] a, double[][] b){
		double[][] result = new double[a.length + b.length][b[0].length];
		for(int i = 0; i < a.length; i++){
			result[i] = a[i];
		}
		for(int i = 0; i < b.length; i++){
			result[i + a.length] = b[i];
		}
		return result;
	}
}
