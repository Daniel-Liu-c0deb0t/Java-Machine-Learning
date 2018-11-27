package javamachinelearning.utils;

import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.awt.image.BufferedImage;

public class ImageUtils {
    private BufferedImage bImage = null;

    public ImageUtils() {

    }

    /*
    Read one color image file and return to 3D int type
    input:
        path : Path of image file
    output:
        3D int array [height][width][3]
     */
    public int[][][] readColorImageFile(String path) {
        File f;
        try {
            f = new File(path);
            bImage = ImageIO.read(f);
        } catch(IOException e) {
            System.out.println("Exception occured : " + e.getMessage());
        }
        int width = bImage.getWidth();
        int height = bImage.getHeight();
        int[][][] data = new int[height][width][3];
        for( int y=0 ; y<height ; y++ ) {
            for( int x = 0 ; x<width ; x++ ) {
                int p = bImage.getRGB(x,y);

                // int a = (p>>24)&0xff;
                int r = (p>>16)&0xff;
                int g = (p>>8)&0xff;
                int b = p&0xff;

                data[y][x][0] = r;
                data[y][x][1] = g;
                data[y][x][2] = b;
            }
        }
        return data;
    }

    /*
    Convert RGB to Gray
     */
    public int[][] convertRGBtoGray(int[][][] colorImg) {
        int height = colorImg.length;
        int width = colorImg[0].length;
        int[][] grayImg = new int[height][width];

        for( int y=0 ; y<height ; y++ ) {
            for( int x=0 ; x<width ; x++ ) {
                // 0.2126*r + 0.7152*g + 0.0722*b;
                grayImg[y][x] = (int)(0.2126*colorImg[y][x][0] + 0.7152*colorImg[y][x][1] + 0.0722*colorImg[y][x][2]);
            }
        }
        return grayImg;
    }


    /*
    Read one image and return to Tensor type
    input:
        path : Image path
    output:
        Tensor type of a image
     */
    public Tensor readColorImageToTensor(String path, boolean convertGray) {
        // Color
        if( convertGray==false ) {
            int[][][] data = readColorImageFile(path);
            return new Tensor(data);
        }
        // Gray
        else {
            int[][][] data = readColorImageFile(path);
            int[][] grayData = convertRGBtoGray(data);
            return new Tensor(grayData);
        }
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
        path : Image folder path
    output:
        Tensor type array of Images
     */
    public Tensor[] readImages(String folderPath, boolean convertGray) {
        File folder = new File(folderPath);
        File[] listOfFiles = folder.listFiles();
        Tensor[] tensors = new Tensor[listOfFiles.length];
            for (int i = 0; i < listOfFiles.length; i++) {
                tensors[i] = readColorImageToTensor(folderPath + listOfFiles[i].getName(), convertGray);
            }
        return tensors;
    }
}
