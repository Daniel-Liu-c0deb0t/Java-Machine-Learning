package graph;

import java.awt.Color;

public class Point{
	private double x, y;
	private Color c;
	
	public Point(double x, double y){
		this.x = x;
		this.y = y;
		this.c = Color.black;
	}
	
	public Point(double x, double y, Color c){
		this.x = x;
		this.y = y;
		this.c = c;
	}
	
	public double getX(){
		return x;
	}
	
	public double getY(){
		return y;
	}
	
	public Color getColor(){
		return c;
	}
}
