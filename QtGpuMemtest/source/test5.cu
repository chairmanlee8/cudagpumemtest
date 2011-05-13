#include "gputests.h"

/************************************************************************************
 * Test 5 [Block move, 64 moves]
 * This test stresses memory by moving block memories. Memory is initialized
 * with shifting patterns that are inverted every 8 bytes.  Then blocks
 * of memory are moved around.  After the moves
 * are completed the data patterns are checked.  Because the data is checked
 * only after the memory moves are completed it is not possible to know
 * where the error occurred.  The addresses reported are only for where the
 * bad pattern was found.
 *
 *
 *************************************************************************************/


__global__ void
kernel_test5_init(char* _ptr, char* end_ptr)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int p1 = 1;
	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i+=16)
	{
		unsigned int p2 = ~p1;

		ptr[i] = p1;
		ptr[i+1] = p1;
		ptr[i+2] = p2;
		ptr[i+3] = p2;
		ptr[i+4] = p1;
		ptr[i+5] = p1;
		ptr[i+6] = p2;
		ptr[i+7] = p2;
		ptr[i+8] = p1;
		ptr[i+9] = p1;
		ptr[i+10] = p2;
		ptr[i+11] = p2;
		ptr[i+12] = p1;
		ptr[i+13] = p1;
		ptr[i+14] = p2;
		ptr[i+15] = p2;

		p1 = p1<<1;
		if (p1 == 0)
		{
			p1 = 1;
		}
	}

	return;
}


__global__ void
kernel_test5_move(char* _ptr, char* end_ptr)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int half_count = BLOCKSIZE/sizeof(unsigned int)/2;
	unsigned int* ptr_mid = ptr + half_count;

	for (i = 0; i < half_count; i++)
	{
		ptr_mid[i] = ptr[i];
	}

	for (i=0; i < half_count - 8; i++)
	{
		ptr[i + 8] = ptr_mid[i];
	}

	for (i=0; i < 8; i++)
	{
		ptr[i] = ptr_mid[half_count - 8 + i];
	}

	return;
}


__global__ void
kernel_test5_check(char* _ptr, char* end_ptr, MemoryError *local_error, int* local_count)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	for (i=0; i < BLOCKSIZE/sizeof(unsigned int); i+=2)
	{
		if (ptr[i] != ptr[i+1])
		{
			record_error(local_error, local_count, &ptr[i], ptr[i+1]);
		}
	}

	return;
}



int
test5(TestInputParams* tip, TestOutputParams* top, bool *term)
{

	unsigned int i;
	char* end_ptr = tip->ptr + tip->tot_num_blocks* BLOCKSIZE;

	for (i=0; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test5_init<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
		//SHOW_PROGRESS("test5[init]", i, tot_num_blocks);
	}


	for (i=0; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test5_move<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
		//SHOW_PROGRESS("test5[move]", i, tot_num_blocks);
	}


	for (i=0; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test5_check<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr, top->err_vector, top->err_count); SYNC_CUERR;
		//error_checking("test5[check]",  i);
		//SHOW_PROGRESS("test5[check]", i, tot_num_blocks);
	}

	return cudaSuccess;

}