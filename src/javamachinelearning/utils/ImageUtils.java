package javamachinelearning.utils;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

public class ImageUtils {
    BufferedImage bImage = null;

    public ImageUtils() {

    }


    /*
    Read one Image and return to Tensor type
    input:
        path : Image path
    output:
        Tensor type of a Image
     */
    public Tensor readOneImage(String path) {
        BufferedImage bIbage = null;
        File f = null;
        try {
            f = new File(path);
            bImage = ImageIO.read(f);
        } catch(IOException e) {
            System.out.println("Exception occured :" + e.getMessage());
        }

        int width = bImage.getWidth();
        int height = bImage.getHeight();

        double[][] data = new double[height][width];

        for( int y=0 ; y<height ; y++ ) {
            for( int x = 0 ; x<width ; x++ ) {
                int p = bImage.getRGB(x,y);

                // int a = (p>>24)&0xff;
                int r = (p>>16)&0xff;
                int g = (p>>8)&0xff;
                int b = p&0xff;

                // Reference = http://entropymine.com/imageworsener/grayscale/
                double gray = 0.2126*r + 0.7152*g + 0.0722*b;
                data[y][x] = gray;
            }
        }
        return new Tensor(data);
    }

    // For testing
    public void readOneImage_Test(String path) {
        BufferedImage bIbage = null;
        File f = null;
        try {
            f = new File(path);
            bImage = ImageIO.read(f);
        } catch(IOException e) {
            System.out.println("Exception occured :" + e.getMessage());
        }

        int width = bImage.getWidth();
        int height = bImage.getHeight();

        for( int y=0 ; y<height ; y++ ) {
            for( int x = 0 ; x<width ; x++ ) {
                int p = bImage.getRGB(x,y);

                // ARGB format
                int a = (p>>24)&0xff;
                int r = (p>>16)&0xff;
                int g = (p>>8)&0xff;
                int b = p&0xff;

                // Reference = http://entropymine.com/imageworsener/grayscale/
                int gray = (int)(0.2126*r + 0.7152*g + 0.0722*b);

                p = (a<<24) | (gray<<16) | (gray<<8) | gray;
                bImage.setRGB(x, y, p);
            }
        }

        try{
            f = new File("test.bmp");
            ImageIO.write(bImage, "bmp", f);
        } catch( IOException e ) {
            System.out.println("Exception occured :" + e.getMessage());
        }
    }

    /*
    Read many Images(in folder) and return to array of Tensor type
    input:
        path : Image path
    output:
        Tensor type array of Images
     */
    public Tensor[] readImages(String path) {
        return null;
    }
}
