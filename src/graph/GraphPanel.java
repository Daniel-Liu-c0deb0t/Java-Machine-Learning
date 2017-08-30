package graph;

import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JPanel;

@SuppressWarnings("serial")
public class GraphPanel extends JPanel{
	private Graph graph;
	
	public GraphPanel(Graph graph){
		this.graph = graph;
		setPreferredSize(new Dimension(graph.getWidth(), graph.getHeight()));
	}
	
	@Override
	public void paintComponent(Graphics graphics){
		super.paintComponent(graphics);
		graphics.drawImage(graph.getGraph(), 0, 0, graph.getWidth(), graph.getHeight(), 0, 0, graph.getWidth(), graph.getHeight(), null);
	}
}
