package javamachinelearning.layers;

public interface RecurrentParamsLayer extends RecurrentLayer, ParamsLayer{
	public RecurrentCell[] cells();
}
