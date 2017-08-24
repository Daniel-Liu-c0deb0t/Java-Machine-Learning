package graph;

import java.awt.Graphics;

import javax.swing.JPanel;

public class GraphPanel extends JPanel{
	private Graph graph;
	
	public GraphPanel(Graph graph){
		this.graph = graph;
	}
	
	@Override
	public void paintComponent(Graphics graphics){
		super.paintComponent(graphics);
		graphics.drawImage(graph.getGraph(), 0, 0, graph.getWidth(), graph.getHeight(), 0, 0, graph.getWidth(), graph.getHeight(), null);
	}
}
