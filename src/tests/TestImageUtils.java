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
        int[][] grayImg = img.convertRGBtoGray(colorImg);
        System.out.println("---Testing converted gray image---");
        System.out.println(grayImg[0][0]);

        // Tensor test(500x480 image)
        System.out.println("---Testing read one image to tensor---");
        Tensor imageTensor = img.readColorImageToTensor("./Images/Set14/baboon.bmp", true);
        System.out.println("Height : " + imageTensor.shape()[1]);
        System.out.println("Width : " + imageTensor.shape()[0]);
        //System.out.println(imageTensor.toString());

        // Operation test and save to "test.bmp"
        img.readOneImage_Test("./Images/Set14/baboon.bmp");

        System.out.println("---Testing read many images to tensors---");
        Tensor[] tensors = img.readImages("./Images/Set14/", true);
        for( int i=0 ; i<tensors.length ; i++ ) {
            System.out.println(tensors[i].shape()[0] + ", " + tensors[i].shape()[1]);
        }
    }
}
