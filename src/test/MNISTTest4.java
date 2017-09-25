package test;

import java.awt.Dimension;
import java.awt.FlowLayout;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;

import draw.DrawablePanel2;
import layer.FCLayer;
import network.SimpleNeuralNetwork;
import utils.Activation;
import utils.UtilMethods;

public class MNISTTest4{
	public static void main(String[] args){
		SimpleNeuralNetwork nn = new SimpleNeuralNetwork(784);
		nn.add(new FCLayer(300, Activation.sigmoid));
		nn.add(new FCLayer(10, Activation.softmax));
		nn.loadFromFile("mnist_weights.nn");
		
		JFrame frame = new JFrame();
		frame.setLayout(new FlowLayout());
		DrawablePanel2 drawablePanel = new DrawablePanel2(1000, 1000, 20, 20);
		frame.add(drawablePanel);
		
		JLabel label = new JLabel("Result: ");
		label.setFont(label.getFont().deriveFont(30.0f));
		frame.add(label);
		
		JButton submitButton = new JButton("Submit");
		submitButton.setPreferredSize(new Dimension(200, 100));
		submitButton.setFont(submitButton.getFont().deriveFont(30.0f));
		submitButton.addActionListener((e) -> {
			double[] data = drawablePanel.getData(28, 28);
			double[] result = nn.predict(data);
			label.setText("Result: " + UtilMethods.argMax(result));
		});
		frame.add(submitButton);
		
		JButton clearButton = new JButton("Clear");
		clearButton.setPreferredSize(new Dimension(200, 100));
		clearButton.setFont(clearButton.getFont().deriveFont(30.0f));
		clearButton.addActionListener((e) -> {
			drawablePanel.clear();
		});
		frame.add(clearButton);
		
		frame.setSize(1200, 1200);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);
	}
}
