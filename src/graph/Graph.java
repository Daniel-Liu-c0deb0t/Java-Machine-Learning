package graph;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import javax.imageio.ImageIO;

import utils.UtilMethods;

public class Graph{
	private BufferedImage graph;
	private Graphics2D graphics;
	private ArrayList<Point> points = new ArrayList<Point>();
	private int width;
	private int height;
	private int xTicks;
	private int yTicks;
	private int padding;
	private String xLabel;
	private String yLabel;
	private ColorFunction colorFunction;
	private boolean customScale = false;
	private double minX;
	private double maxX;
	private double minY;
	private double maxY;
	
	public Graph(ColorFunction colorFunction){
		this(500, 500, colorFunction);
	}
	
	public Graph(int width, int height, ColorFunction colorFunction){
		this(width, height, null, null, null, colorFunction);
	}
	
	public Graph(int width, int height, double[] xData, double[] yData, Color[] cData, ColorFunction colorFunction){
		this(width, height, "x-axis", "y-axis", xData, yData, cData, colorFunction);
	}
	
	public Graph(int width, int height, String xLabel, String yLabel, double[] xData, double[] yData, Color[] cData, ColorFunction colorFunction){
		this(width, height, 10, 10, 100, xLabel, yLabel, xData, yData, cData, colorFunction);
	}
	
	public Graph(int width, int height, int xTicks, int yTicks, int padding, String xLabel, String yLabel, double[] xData, double[] yData, Color[] cData, ColorFunction colorFunction){
		this.graph = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
		this.graphics = this.graph.createGraphics();
		this.width = width;
		this.height = height;
		this.xTicks = xTicks + 1;
		this.yTicks = yTicks + 1;
		this.padding = padding;
		this.xLabel = xLabel;
		this.yLabel = yLabel;
		this.colorFunction = colorFunction;
		
		if(xData != null && yData != null){
			for(int i = 0; i < xData.length; i++){
				if(cData == null || cData.length < xData.length)
					this.points.add(new Point(xData[i], yData[i]));
				else
					this.points.add(new Point(xData[i], yData[i], cData[i]));
			}
		}
	}
	
	public void useCustomScale(double minX, double maxX, double minY, double maxY){
		this.minX = minX;
		this.maxX = maxX;
		this.minY = minY;
		this.maxY = maxY;
		this.customScale = true;
	}
	
	public void usePointScale(){
		this.customScale = false;
	}
	
	public void draw(){
		//find graph range
		double xMax = Double.MIN_VALUE;
		double xMin = Double.MAX_VALUE;
		double yMax = Double.MIN_VALUE;
		double yMin = Double.MAX_VALUE;
		if(customScale){
			xMax = maxX;
			xMin = minX;
			yMax = maxY;
			yMin = minY;
		}else{
			for(int i = 0; i < points.size(); i++){
				xMax = Math.max(xMax, points.get(i).getX());
				xMin = Math.min(xMin, points.get(i).getX());
				yMax = Math.max(yMax, points.get(i).getY());
				yMin = Math.min(yMin, points.get(i).getY());
			}
			if(xMax == Double.MIN_VALUE)
				xMax = 10;
			if(xMin == Double.MAX_VALUE)
				xMin = 0;
			if(yMax == Double.MIN_VALUE)
				yMax = 10;
			if(yMin == Double.MAX_VALUE)
				yMin = 0;
		}
		
		if(colorFunction != null){
			int xSize = 500;
			int ySize = 500;
			for(int i = 0; i < xSize; i++){
				for(int j = 0; j < ySize; j++){
					graphics.setColor(colorFunction.getColor(xMin + i / (double)xSize * (xMax - xMin), yMin + j / (double)ySize * (yMax - yMin)));
					graphics.fillRect((int)(padding * 2 + i * (width - padding * 3) / (double)xSize), (int)(height - padding * 2 - (j + 1) * ((height - padding * 3) / (double)ySize)), (int)((width - padding * 3) / (double)xSize), (int)((height - padding * 3) / (double)ySize));
				}
			}
		}
		
		//x and y axis
		graphics.setColor(Color.black);
		graphics.setStroke(new BasicStroke(3));
		graphics.drawLine(padding * 2, height - padding * 2, width - padding, height - padding * 2);
		graphics.drawLine(padding * 2, height - padding * 2, padding * 2, padding);
		
		//x and y axis labels
		graphics.setFont(graphics.getFont().deriveFont(50.0f));
		graphics.drawString(xLabel, width / 2 - 5 * xLabel.length(), height - padding);
		AffineTransform oldTransform = graphics.getTransform();
		graphics.translate(padding, height / 2 + 5 * yLabel.length());
		graphics.rotate(Math.toRadians(-90.0));
		graphics.drawString(yLabel, 0, 0);
		graphics.setTransform(oldTransform);
		
		//draw tick marks
		graphics.setFont(graphics.getFont().deriveFont(25.0f));
		int xTickSpacing = (width - padding * 3) / (xTicks - 1);
		int yTickSpacing = (height - padding * 3) / (yTicks - 1);
		for(int i = 0; i < xTicks; i++){
			graphics.drawLine(padding * 2 + xTickSpacing * i, height - padding * 2, padding * 2 + xTickSpacing * i, height - padding * 2 + 10);
			graphics.drawString(UtilMethods.shorterFormat(xMin + (xMax - xMin) / (xTicks - 1) * i), padding * 2 + xTickSpacing * i - 7, height - padding * 2 + 40);
		}
		for(int i = 0; i < yTicks; i++){
			graphics.drawLine(padding * 2, height - padding * 2 - yTickSpacing * i, padding * 2 - 10, height - padding * 2 - yTickSpacing * i);
			graphics.drawString(UtilMethods.shorterFormat(yMin + (yMax - yMin) / (yTicks - 1) * i), padding * 2 - 70, height - padding * 2 - yTickSpacing * i + 10);
		}
		
		//draw points
		for(int i = 0; i < points.size(); i++){
			graphics.setColor(points.get(i).getColor());
			graphics.fillOval(padding * 2 + (int)(points.get(i).getX() / (xMax - xMin) * (width - padding * 3)) - 8, (height - padding * 2) - (int)(points.get(i).getY() / (yMax - yMin) * (height - padding * 3)) - 8, 16, 16);
		}
	}
	
	public void saveToFile(String path, String type){
		try{
			ImageIO.write(graph, type, new File(path));
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	
	public void addPoint(double x, double y, Color c){
		points.add(new Point(x, y, c));
	}
	
	public void addPoint(double x, double y){
		points.add(new Point(x, y));
	}
	
	public BufferedImage getGraph(){
		return graph;
	}
	
	public int getWidth(){
		return width;
	}
	
	public int getHeight(){
		return height;
	}
	
	public void dispose(){
		graphics.dispose();
	}
}
