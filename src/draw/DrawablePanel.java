package draw;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;

import javax.swing.JPanel;

@SuppressWarnings("serial")
public class DrawablePanel extends JPanel{
	private int width;
	private int height;
	private int xSize;
	private int ySize;
	private BufferedImage image;
	private Graphics2D graphics;
	private double[][] arr;
	
	public DrawablePanel(int width, int height, int xSize, int ySize){
		this.width = width;
		this.height = height;
		this.xSize = xSize;
		this.ySize = ySize;
		setPreferredSize(new Dimension(width, height));
		this.image = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_ARGB);
		this.arr = new double[xSize][ySize];
		graphics = this.image.createGraphics();
		graphics.setColor(Color.black);
		addMouseMotionListener(new MouseMotionListener(){
			@Override
			public void mouseDragged(MouseEvent e){
				if(e.getX() / (width / xSize) < arr.length && e.getX() / (width / xSize) >= 0 && e.getY() / (height / ySize) < arr[0].length && e.getY() / (height / ySize) >= 0){
					graphics.fillRect(e.getX() / (width / xSize), e.getY() / (height / ySize), 1, 1);
					arr[e.getX() / (width / xSize)][e.getY() / (height / ySize)] = 1.0;
				}
				repaint();
			}
			
			@Override
			public void mouseMoved(MouseEvent e){
				
			}
		});
	}
	
	public double[][] getData(){
		return arr;
	}
	
	public void clear(){
		image = new BufferedImage(xSize, ySize, BufferedImage.TYPE_INT_ARGB);
		graphics = image.createGraphics();
		graphics.setColor(Color.black);
		for(int i = 0; i < xSize; i++){
			for(int j = 0; j < ySize; j++){
				arr[i][j] = 0.0;
			}
		}
		repaint();
	}
	
	@Override
	public void paintComponent(Graphics g){
		super.paintComponent(g);
		g.drawImage(image, 0, 0, width * width / xSize, height * height / ySize, 0, 0, width, height, null);
	}
}
