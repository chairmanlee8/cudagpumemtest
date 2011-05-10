#include "gputests.h"

int
test2(char* ptr, unsigned int tot_num_blocks, int num_iterations, unsigned int* err, unsigned long* err_addr,
      unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read, bool *term)
{
	unsigned int p1 = 0;
	unsigned int p2 = ~p1;


	//DEBUG_PRINTF("Test2: Moving inversions test, with pattern 0x%x and 0x%x\n", p1, p2);
	move_inv_test(ptr, tot_num_blocks, p1, p2, err, err_addr, err_expect, err_current, err_second_read, term);
	//DEBUG_PRINTF("Test2: Moving inversions test, with pattern 0x%x and 0x%x\n", p2, p1);
	move_inv_test(ptr, tot_num_blocks, p2, p1, err, err_addr, err_expect, err_current, err_second_read, term);

	return cudaSuccess;

}

