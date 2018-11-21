package tests;

import javamachinelearning.utils.ImageUtils;
import javamachinelearning.utils.Tensor;

public class TestImageUtils {
    public static void main(String args[]) {
        ImageUtils img = new ImageUtils();
        // Tensor test(500x480 image)
        Tensor imageTensor = img.readOneImage("./Images/Set14/baboon.bmp");
        System.out.println("Height : " + imageTensor.shape()[0]);
        System.out.println("Width : " + imageTensor.shape()[1]);
        //System.out.println(imageTensor.toString());

        // Operation test and save to "test.bmp"
        img.readOneImage_Test("./Images/Set14/baboon.bmp");
    }
}
