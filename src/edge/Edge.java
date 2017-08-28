package edge;

import java.util.Random;

public class Edge{
	private static final Random rand = new Random();
	private double weight;
	private double deltaWeight;
	private int nodeA;
	private int nodeB;
	
	public Edge(int nodeA, int nodeB, int prevSize){
		this.nodeA = nodeA;
		this.nodeB = nodeB;
		this.weight = rand.nextGaussian() / Math.sqrt(prevSize);
	}
	
	public Edge(int nodeA, int nodeB, double weight){
		this.nodeA = nodeA;
		this.nodeB = nodeB;
		this.weight = weight;
	}
	
	public double getWeight(){
		return weight;
	}
	
	public void setWeight(double w){
		this.weight = w;
	}
	
	public void addWeight(double w){
		this.deltaWeight += w;
	}
	
	public void update(){
		this.weight += this.deltaWeight;
		this.deltaWeight = 0.0;
	}
	
	public int getNodeA(){
		return nodeA;
	}
	
	public void setNodeA(int nodeA){
		this.nodeA = nodeA;
	}
	
	public int getNodeB(){
		return nodeB;
	}
	
	public void setNodeB(int nodeB){
		this.nodeB = nodeB;
	}
}
