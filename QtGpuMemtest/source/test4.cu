#include "gputests.h"

/************************************************************************************
 * Test 4 [Moving inversions, random pattern]
 * Test 4 uses the same algorithm as test 1 but the data pattern is a
 * random number and it's complement. This test is particularly effective
 * in finding difficult to detect data sensitive errors. A total of 60
 * patterns are used. The random number sequence is different with each pass
 * so multiple passes increase effectiveness.
 *
 *************************************************************************************/

int
test4(TestInputParams* tip, TestOutputParams* top, bool *term)
{
	unsigned int p1;
	//if (global_pattern == 0){
	p1 = get_random_num();
	//}else{
	//p1 = global_pattern;
	//}

	unsigned int p2 = ~p1;
	unsigned int err = 0;
	unsigned int iteration = 0;

	//DEBUG_PRINTF("Test4: Moving inversions test, with random pattern 0x%x and 0x%x\n", p1, p2);

repeat:
	err += move_inv_test(tip->ptr, tip->tot_num_blocks, p1, p2, top->err_vector, top->err_count, term);

	if (err == 0 && iteration == 0)
	{
		return cudaSuccess;
	}
	if (iteration < tip->num_iterations)
	{
		//if(*term == true) break;
		//PRINTF("%dth repeating test4 because there are %d errors found in last run\n", iteration, err);
		iteration++;
		err = 0;
		if(*term == false) goto repeat;
	}

	return cudaSuccess;
}