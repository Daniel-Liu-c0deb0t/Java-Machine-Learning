package tests;

import java.awt.Dimension;
import java.awt.FlowLayout;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;

import javamachinelearning.drawables.MNISTDrawablePanel2;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.TensorUtils;

public class TestMNISTDraw2{
	public static void main(String[] args){
		// make sure the drawings are big enough!
		
		SequentialNN nn = new SequentialNN(784);
		nn.add(new FCLayer(300, Activation.relu));
		nn.add(new FCLayer(10, Activation.softmax));
		nn.loadFromFile("mnist_weights_fc.nn");
		
		JFrame frame = new JFrame();
		frame.setLayout(new FlowLayout());
		MNISTDrawablePanel2 drawablePanel = new MNISTDrawablePanel2(1000, 1000, 20, 20);
		frame.add(drawablePanel);
		
		JLabel label = new JLabel("Result: ");
		label.setFont(label.getFont().deriveFont(30.0f));
		frame.add(label);
		
		JButton submitButton = new JButton("Submit");
		submitButton.setPreferredSize(new Dimension(200, 100));
		submitButton.setFont(submitButton.getFont().deriveFont(30.0f));
		submitButton.addActionListener((e) -> {
			Tensor data = drawablePanel.getData(28, 28);
			Tensor result = nn.predict(data.flatten());
			label.setText("Result: " + TensorUtils.argMax(result));
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
