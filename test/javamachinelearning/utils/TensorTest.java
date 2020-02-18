package javamachinelearning.utils;


import static javamachinelearning.utils.TensorUtils.t;
import static org.junit.Assert.*;

import org.junit.Test;

public class TensorTest {
	@Test
	public void when_add_two_tensor_res_is_3() {
		double[] data1 = { 1.0 };
		double[] data2 = { 2.0 };
		
		Tensor a = new Tensor(data1);
		Tensor b = new Tensor(data2);
		Tensor res = a.add(b);
		
		assertEquals("result was not correct.", 3.0, res.flatGet(0), 0);
	}
}
