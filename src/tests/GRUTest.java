package tests;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

import javamachinelearning.layers.feedforward.ActivationLayer;
import javamachinelearning.layers.feedforward.FCLayer;
import javamachinelearning.layers.feedforward.ScalingLayer;
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
		String string = "hello! what is your name? i am a recurrent neural network!";
		string = Utils.pad(string, 60, ' ');
		string = string + " ";
		
		int epochs = 100;
		int batchSize = 10;
		int winSize = 10;
		int winStep = 10;
		int genIter = 200;
		double temperature = 0.1;
		
		SequentialNN nn = new SequentialNN(winSize, alphabet.length());
		nn.add(new RecurrentLayer(winSize, new GRUCell(), true));
		nn.add(new RecurrentLayer(winSize, new GRUCell(), true));
		nn.add(new FCLayer(alphabet.length()));
		nn.add(new ScalingLayer(1 / temperature, false));
		nn.add(new ActivationLayer(Activation.softmax));
		
		String[] str = Utils.slide(string, winSize);
		
		ArrayList<Tensor> xArr = new ArrayList<>();
		ArrayList<Tensor> tArr = new ArrayList<>();
		for(int i = 0; i < str.length - 1; i += winStep){
			xArr.add(TensorUtils.oneHotString(str[i], alphabet));
			tArr.add(TensorUtils.oneHotString(str[i + 1], alphabet));
		}
		
		Tensor[] xs = xArr.toArray(new Tensor[0]);
		Tensor[] ts = tArr.toArray(new Tensor[0]);
		
		nn.train(xs,
				ts,
				epochs,
				batchSize,
				Loss.softmaxCrossEntropy,
				new AdamOptimizer(0.01),
				null,
				false,
				false,
				(epoch, error) -> nn.resetStates());
		
		System.out.print("Input seed string: ");
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String seed = r.readLine();
		r.close();
		
		StringBuilder gen = new StringBuilder();
		gen.append(seed);
		
		Tensor seedInput = TensorUtils.oneHotString(seed, alphabet);
		nn.predict(seedInput, seed.length());
		
		for(int i = 0; i < genIter; i++){
			Tensor inputStr = TensorUtils.oneHotString(gen.charAt(gen.length() - 1) + "", alphabet);
			String outputStr = TensorUtils.decodeString(nn.predict(inputStr, 1), true, alphabet);
			gen.append(outputStr);
		}
		
		System.out.println("Output: " + gen.toString());
	}
}
