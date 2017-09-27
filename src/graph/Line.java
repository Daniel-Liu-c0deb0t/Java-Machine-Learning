package graph;

import java.awt.Color;

public class Line{
	private double m, b;
	private Color c;
	
	public Line(double m, double b){
		this.m = m;
		this.b = b;
		this.c = Color.black;
	}
	
	public Line(double m, double b, Color c){
		this.m = m;
		this.b = b;
		this.c = c;
	}
	
	public double getM(){
		return m;
	}
	
	public double getB(){
		return b;
	}
	
	public Color getColor(){
		return c;
	}
}
