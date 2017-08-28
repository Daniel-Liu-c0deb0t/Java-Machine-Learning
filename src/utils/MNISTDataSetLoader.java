package utils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

public class MNISTDataSetLoader{
	public static double[][] loadImages(String path, int num){
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
	
	public static double[][] loadLabels(String path, int num){
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
}
