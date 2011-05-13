#include "gputests.h"

int
test3(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	unsigned int p0=0x80;
	unsigned int p1 = p0 | (p0 << 8) | (p0 << 16) | (p0 << 24);
	unsigned int p2 = ~p1;

	//DEBUG_PRINTF("Test3: Moving inversions test, with pattern 0x%x and 0x%x\n", p1, p2);
	move_inv_test(tip->ptr, tip->tot_num_blocks, p1, p2, top->err_vector, top->err_count, term);
	//DEBUG_PRINTF("Test3: Moving inversions test, with pattern 0x%x and 0x%x\n", p2, p1);
	move_inv_test(tip->ptr, tip->tot_num_blocks, p2, p1, top->err_vector, top->err_count, term);

	return cudaSuccess;

}