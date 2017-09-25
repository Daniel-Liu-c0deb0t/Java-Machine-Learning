package draw;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

import utils.UtilMethods;

@SuppressWarnings("serial")
public class DrawablePanel2 extends JPanel{
	private int width;
	private int height;
	private int xSize;
	private int ySize;
	private BufferedImage image;
	private Graphics2D graphics;
	
	public DrawablePanel2(int width, int height, int xSize, int ySize){
		this.width = width;
		this.height = height;
		this.xSize = xSize;
		this.ySize = ySize;
		setPreferredSize(new Dimension(width, height));
		this.image = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_RGB);
		graphics = this.image.createGraphics();
		graphics.setColor(Color.white);
		graphics.fillRect(0, 0, xSize, ySize);
		graphics.setColor(Color.black);
		addMouseMotionListener(new MouseMotionListener(){
			@Override
			public void mouseDragged(MouseEvent e){
				graphics.fillRect((int)((double)e.getX() / ((double)width / (double)xSize)), (int)((double)e.getY() / ((double)height / (double)ySize)), 1, 1);
				repaint();
			}
			
			@Override
			public void mouseMoved(MouseEvent e){
				
			}
		});
	}
	
	public double[] getData(int outputX2, int outputY2){
		double[][] arr = new double[xSize][ySize];
		for(int i = 0; i < xSize; i++){
			for(int j = 0; j < ySize; j++){
				arr[i][j] = 1.0 - (image.getRGB(i, j) & 0xFF) / 255.0;
			}
		}
		return UtilMethods.flattenData(UtilMethods.centerData(arr, outputX2, outputY2));
	}
	
	public void clear(){
		image = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_RGB);
		graphics = image.createGraphics();
		graphics.setColor(Color.white);
		graphics.fillRect(0, 0, xSize, ySize);
		graphics.setColor(Color.black);
		repaint();
	}
	
	@Override
	public void paintComponent(Graphics g){
		super.paintComponent(g);
		g.drawImage(image, 0, 0, width * width / xSize, height * height / ySize, 0, 0, width, height, null);
	}
}
