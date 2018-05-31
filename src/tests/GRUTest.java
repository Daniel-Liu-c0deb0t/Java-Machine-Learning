package tests;

import java.io.BufferedReader;
import java.io.InputStreamReader;

import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.layers.recurrent.GRUCell;
import javamachinelearning.layers.recurrent.RecurrentLayer;
import javamachinelearning.networks.SequentialNN;
import javamachinelearning.optimizers.AdamOptimizer;
import javamachinelearning.utils.Activation;
import javamachinelearning.utils.Loss;
import javamachinelearning.utils.Tensor;
import javamachinelearning.utils.TensorUtils;
import javamachinelearning.utils.Utils;

public class GRUTest{
	public static void main(String[] args) throws Exception{
		String alphabet = "abcdefghijklmnopqrstuvwxyz .,?!";
		String string = "hello world";
		int winSize = 5;
		int genIter = 100;
		
		SequentialNN nn = new SequentialNN(winSize, alphabet.length());
		nn.add(new RecurrentLayer(winSize, new GRUCell()));
		nn.add(new RecurrentLayer(winSize, new GRUCell()));
		nn.add(new FCLayer(alphabet.length(), Activation.softmax));
		
		String[] str = Utils.slide(string, winSize);
		
		Tensor[] xs = new Tensor[str.length - 1];
		Tensor[] ts = new Tensor[str.length - 1];
		for(int i = 0; i < str.length - 1; i++){
			xs[i] = TensorUtils.oneHotString(str[i], alphabet);
			ts[i] = TensorUtils.oneHotString(str[i + 1], alphabet);
		}
		
		nn.train(xs, ts, 100, 10, Loss.softmaxCrossEntropy, new AdamOptimizer(0.01), null, false, false, false);
		
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String seed = r.readLine();
		r.close();
		
		StringBuilder gen = new StringBuilder();
		gen.append(seed);
		
		for(int i = 0; i < genIter; i++){
			Tensor inputStr = TensorUtils.oneHotString(
					gen.substring(Math.max(gen.length() - winSize, 0)), alphabet);
			
			String outputStr = TensorUtils.decodeString(
					nn.predict(inputStr), false, alphabet);
			
			gen.append(outputStr.charAt(
					outputStr.length() - 1 + Math.min(gen.length() - winSize, 0)));
		}
		
		System.out.println(gen.toString());
	}
}
