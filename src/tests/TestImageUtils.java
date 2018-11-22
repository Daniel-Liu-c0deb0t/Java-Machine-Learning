package tests;

import javamachinelearning.utils.ImageUtils;
import javamachinelearning.utils.Tensor;

public class TestImageUtils {
    public static void main(String args[]) {
        ImageUtils img = new ImageUtils();

        // test readColorImageFile
        int[][][] colorImg = img.readColorImageFile("./Images/Set14/foreman.bmp");
        System.out.print(colorImg[0][0][0] + ", ");
        System.out.print(colorImg[0][0][1] + ", ");
        System.out.println(colorImg[0][0][2]);

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
