package tests;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;

import javamachinelearning.layers.feedforward.ActivationLayer;
import javamachinelearning.layers.feedforward.DropoutLayer;
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
		// all of the letters that can appear in the text
		String alphabet = "abcdefghijklmnopqrstuvwxyz .,?!\n'()-";
		boolean readFromFile = true;
		
		// can optionally read a (long) string from a file
		String string;
		if(readFromFile)
			string = new String(Files.readAllBytes(Paths.get("rnn_training_romeo_juliet.txt")));
		else
			string = "hello! what is your name? i am a recurrent neural network!";
		
		// preprocess the string
		string = string.toLowerCase();
		string = string.replace("\r\n", "\n");
		// remove characters that are not found in the alphabet
		string = Utils.onlyKeepAlphabetChars(string, alphabet);
		
		int epochs = 500;
		int batchSize = 10;
		int winSize = 20;
		int winStep = 20; // winSize = winStep so substrings are not repeated
		int genIter = 5000; // how many characters to generate
		double temperature = 0.1; // lower = less randomness
		
		// pad the string with spaces to make it divisible by winStep
		string = Utils.pad(string, (int)Math.ceil((double)string.length() / winStep) * winStep + 1, ' ');
		
		// builds the network
		// for each time step, the input is a one hot vector describing the current character
		// for each time step, the output is a one hot vector describing the next character
		// the recurrent layers are stateful, which means that the next state relies on the previous states
		SequentialNN nn = new SequentialNN(winSize, alphabet.length());
		nn.add(new RecurrentLayer(winSize, new GRUCell(), true));
		nn.add(new DropoutLayer(0.3));
		nn.add(new RecurrentLayer(winSize, new GRUCell(), true));
		// the same fully connected layer is applied for every single time step
		nn.add(new FCLayer(alphabet.length()));
		// scales the values by the temperature before softmax
		nn.add(new ScalingLayer(1 / temperature, false));
		nn.add(new ActivationLayer(Activation.softmax));
		
		// get all substrings
		String[] str = Utils.slide(string, winSize);
		
		// skip some substrings if needed and one hot the strings
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
				null, // no regularization
				false, // no shuffling!
				false,
				(epoch, error) -> nn.resetStates()); // reset the GRU cell states every epoch
		
		// reads the seed string
		System.out.print("Input seed string: ");
		BufferedReader r = new BufferedReader(new InputStreamReader(System.in));
		String seed = r.readLine();
		r.close();
		
		StringBuilder gen = new StringBuilder();
		gen.append(seed);
		
		// warms up the model with the seed string
		if(seed.length() > 1){
			Tensor seedInput = TensorUtils.oneHotString(seed.substring(0, seed.length() - 1), alphabet);
			nn.predict(seedInput, seed.length() - 1);
		}
		
		// for each iteration, the previous character is plugged in as one time step
		// and the next character is predicted
		// the previous states persists throughout the entire generation process
		for(int i = 0; i < genIter; i++){
			Tensor inputStr = TensorUtils.oneHotString(gen.charAt(gen.length() - 1) + "", alphabet);
			String outputStr = TensorUtils.decodeString(nn.predict(inputStr, 1), true, alphabet);
			gen.append(outputStr);
		}
		
		System.out.println("Output: " + gen.toString());
	}
}
