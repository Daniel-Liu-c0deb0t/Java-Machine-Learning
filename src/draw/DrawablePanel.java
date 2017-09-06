package draw;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

import utils.UtilMethods;

@SuppressWarnings("serial")
public class DrawablePanel extends JPanel{
	private int width;
	private int height;
	private int xSize;
	private int ySize;
	private BufferedImage image;
	private Graphics2D graphics;
	
	public DrawablePanel(int width, int height, int xSize, int ySize){
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
				graphics.fillRect((int)((double)e.getX() / ((double)width / (double)xSize)), (int)((double)e.getY() / ((double)height / (double)ySize)), 2, 2);
				repaint();
			}
			
			@Override
			public void mouseMoved(MouseEvent e){
				
			}
		});
	}
	
	public double[] getData(int outputX1, int outputY1, int outputX2, int outputY2){
		int minX = Integer.MAX_VALUE;
		int maxX = 0;
		int minY = Integer.MAX_VALUE;
		int maxY = 0;
		for(int i = 0; i < xSize; i++){
			for(int j = 0; j < ySize; j++){
				if((image.getRGB(i, j) & 0xFF) < 255){
					minX = Math.min(minX, i);
					maxX = Math.max(maxX, i);
					minY = Math.min(minY, j);
					maxY = Math.max(maxY, j);
				}
			}
		}
		if((maxX - minX) * 3 < maxY - minY){
			minX -= (maxX - minX) * 1.5;
			maxX += (maxX - minX) * 1.5;
		}
		BufferedImage temp = new BufferedImage(maxX - minX, maxY - minY, image.getType());
		Graphics2D g = temp.createGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, maxX - minX, maxY - minY);
		g.dispose();
		for(int i = minX; i < maxX; i++){
			for(int j = minY; j < maxY; j++){
				if(i >= 0 && i < xSize && j >= 0 && j < ySize){
					temp.setRGB(i - minX, j - minY, image.getRGB(i, j));
				}
			}
		}
		
		BufferedImage result = new BufferedImage(outputX1, outputY1, image.getType());
		g = result.createGraphics();
		g.setColor(Color.white);
		g.fillRect(0, 0, outputX1, outputY1);
		g.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC);
		g.drawImage(temp, 0, 0, outputX1, outputY1, 0, 0, temp.getWidth(), temp.getHeight(), null);
		g.dispose();
		double[][] arr = new double[outputX1][outputY1];
		for(int i = 0; i < outputX1; i++){
			for(int j = 0; j < outputY1; j++){
				arr[i][j] = 1.0 - (result.getRGB(i, j) & 0xFF) / 255.0;
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
