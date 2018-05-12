package utils;

import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.imageio.ImageIO;

public class MNISTUtils{
	public static Tensor[] loadDataSetImages(String path, int num){
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
		Tensor[] res = new Tensor[Math.min(count, num)];
		for(int i = 0; i < Math.min(count, num); i++){
			double[] curr = new double[row * col];
			for(int j = 0; j < row * col; j++){
				curr[j] = UtilMethods.unsignedByteToInt(bb.get()) / 255.0;
			}
			res[i] = new Tensor(curr).reshape(new int[]{row, col}).T();
		}
		return res;
	}
	
	public static Tensor[] loadDataSetLabels(String path, int num){
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
		Tensor[] res = new Tensor[Math.min(count, num)];
		for(int i = 0; i < Math.min(count, num); i++){
			res[i] = UtilMethods.oneHotEncode(UtilMethods.unsignedByteToInt(bb.get()), 10);
		}
		return res;
	}
	
	public static Tensor loadImage(String path, int width, int height){
		BufferedImage image = null;
		try{
			image = ImageIO.read(new File(path));
		}catch(Exception e){
			e.printStackTrace();
		}
		return loadImage(image, width, height);
	}
	
	public static Tensor loadImage(BufferedImage image, int width, int height){
		double[][] arr = new double[image.getWidth()][image.getHeight()];
		for(int i = 0; i < image.getWidth(); i++){
			for(int j = 0; j < image.getHeight(); j++){
				arr[i][j] = 1.0 - ((image.getRGB(i, j) & 0xFF) / 255.0);
			}
		}
		
		return UtilMethods.centerData(arr, width, height);
	}
}
