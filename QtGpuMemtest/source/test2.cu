#include "gputests.h"

int
test2(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	unsigned int p1 = 0;
	unsigned int p2 = ~p1;


	//DEBUG_PRINTF("Test2: Moving inversions test, with pattern 0x%x and 0x%x\n", p1, p2);
	move_inv_test(tip->ptr, tip->tot_num_blocks, p1, p2, top->err_vector, top->err_count, term);
	//DEBUG_PRINTF("Test2: Moving inversions test, with pattern 0x%x and 0x%x\n", p2, p1);
	move_inv_test(tip->ptr, tip->tot_num_blocks, p2, p1, top->err_vector, top->err_count, term);

	return cudaSuccess;

}

