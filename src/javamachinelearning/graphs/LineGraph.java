package javamachinelearning.graphs;

import java.awt.Color;
import java.util.ArrayList;

public class LineGraph{
	private ArrayList<Point> arr;
	private Color c;
	
	public LineGraph(ArrayList<Point> arr){
		this.arr = arr;
		this.c = Color.black;
	}
	
	public LineGraph(ArrayList<Point> arr, Color c){
		this.arr = arr;
		this.c = c;
	}
	
	public ArrayList<Point> getPoints(){
		return arr;
	}
	
	public Color getColor(){
		return c;
	}
}
