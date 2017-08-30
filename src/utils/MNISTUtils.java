package utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

public class MNISTUtils{
	public static double[][] loadDataSetImages(String path, int num){
		byte[] bytes = null;
		try{
			bytes = Files.readAllBytes(Paths.get(path));
		}catch(Exception e){
			e.printStackTrace();
		}
		ByteBuffer bb = ByteBuffer.wrap(bytes);
		bb.order(ByteOrder.BIG_ENDIAN);
		int count = bb.getInt(4);
		int row = bb.getInt(8);
		int col = bb.getInt(12);
		bb.position(16);
		double[][] result = new double[Math.min(count, num)][row * col];
		for(int i = 0; i < Math.min(count, num); i++){
			for(int j = 0; j < row * col; j++){
				result[i][j] = UtilMethods.unsignedByteToInt(bb.get()) / 255.0;
			}
		}
		return result;
	}
	
	public static double[][] loadDataSetLabels(String path, int num){
		byte[] bytes = null;
		try{
			bytes = Files.readAllBytes(Paths.get(path));
		}catch(Exception e){
			e.printStackTrace();
		}
		ByteBuffer bb = ByteBuffer.wrap(bytes);
		bb.order(ByteOrder.BIG_ENDIAN);
		int count = bb.getInt(4);
		bb.position(8);
		double[][] result = new double[Math.min(count, num)][10];
		for(int i = 0; i < Math.min(count, num); i++){
			result[i] = UtilMethods.oneHotEncode(UtilMethods.unsignedByteToInt(bb.get()), 10);
		}
		return result;
	}
	
	public static double[] loadImage(String path, int width, int height){
		BufferedImage image = null;
		try{
			image = ImageIO.read(new File(path));
		}catch(Exception e){
			e.printStackTrace();
		}
		return loadImage(image, width, height);
	}
	
	public static double[] loadImage(BufferedImage image, int width, int height){
		double[][] arr = new double[image.getWidth()][image.getHeight()];
		for(int i = 0; i < image.getWidth(); i++){
			for(int j = 0; j < image.getHeight(); j++){
				arr[i][j] = 1.0 - ((image.getRGB(i, j) & 0xFF) / 255.0);
			}
		}
		
		return centerData(arr, width, height);
	}
	
	public static double[] centerData(double[][] arr, int width, int height){
		double[][] centeredArr = new double[width][height];
		int[] centerOfMass = UtilMethods.centerOfMass(arr);
		for(int i = 0; i < arr.length; i++){
			for(int j = 0; j < arr[i].length; j++){
				if(arr[i][j] > 0.0)
					centeredArr[(width - arr.length) / 2 + i + arr.length / 2 - centerOfMass[0]][(height - arr[i].length) / 2 + j + arr[i].length / 2 - centerOfMass[1]] = arr[i][j];
			}
		}
		double[] result = new double[width * height];
		for(int i = 0; i < height; i++){
			for(int j = 0; j < width; j++){
				result[i * width + j] = centeredArr[j][i];
			}
		}
		return result;
	}
}
