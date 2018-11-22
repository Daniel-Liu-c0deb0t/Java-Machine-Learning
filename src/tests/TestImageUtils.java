package tests;

import javamachinelearning.utils.ImageUtils;
import javamachinelearning.utils.Tensor;

public class TestImageUtils {
    public static void main(String args[]) {
        ImageUtils img = new ImageUtils();

        // Test readColorImageFile
        int[][][] colorImg = img.readColorImageFile("./Images/Set14/comic.bmp");
        System.out.println("---Testing color image---");
        System.out.print(colorImg[0][0][0] + ", ");
        System.out.print(colorImg[0][0][1] + ", ");
        System.out.println(colorImg[0][0][2]);

        // Test convertRGBtoGray
        int[][] grayImg = img.covertRGBtoGray(colorImg);
        System.out.println("---Testing converted gray image---");
        System.out.println(grayImg[0][0]);

        // Tensor test(500x480 image)
        Tensor imageTensor = img.readOneImage("./Images/Set14/baboon.bmp");
        System.out.println("Height : " + imageTensor.shape()[1]);
        System.out.println("Width : " + imageTensor.shape()[0]);
        //System.out.println(imageTensor.toString());

        // Operation test and save to "test.bmp"
        img.readOneImage_Test("./Images/Set14/baboon.bmp");

        //img.readImages("./Images/Set14/");
    }
}
