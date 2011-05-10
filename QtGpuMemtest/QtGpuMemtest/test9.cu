#include "gputests.h"

/************************************************************************************
 *
 * Test 9 [Bit fade test, 90 min, 2 patterns]
 * The bit fade test initializes all of memory with a pattern and then
 * sleeps for 90 minutes. Then memory is examined to see if any memory bits
 * have changed. All ones and all zero patterns are used. This test takes
 * 3 hours to complete.  The Bit Fade test is disabled by default
 *
 **********************************************************************************/

__global__ extern void kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern);
__global__ extern void kernel_move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* err, unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);
__global__ extern void kernel_move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, unsigned int* err, unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read );

int
test9(char* ptr, unsigned int tot_num_blocks, int num_iterations, unsigned int* err_count, unsigned long* err_addr,
      unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read, bool *term)
{

	unsigned int p1 = 0;
	unsigned int p2 = ~p1;

	unsigned int i;
	char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

	for (i= 0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_move_inv_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p1); SYNC_CUERR;
		//SHOW_PROGRESS("test9[bit fade test, write]", i, tot_num_blocks);
	}

	//DEBUG_PRINTF("sleeping for 90 minutes\n");
	//sleep(60*90);
	//Sleep(60*90*1000);
	for(i = 0; i < 1000*9; i++)
	{
		Sleep(60*10);
		if(*term == true) break;
	}

	for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_move_inv_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p1, p2, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
		//error_checking("test9[bit fade test, readwrite]",  i);
		//SHOW_PROGRESS("test9[bit fade test, readwrite]", i, tot_num_blocks);
	}

	//DEBUG_PRINTF("sleeping for 90 minutes\n");
	//sleep(60*90);
	//Sleep(60*90*1000);
	for(i = 0; i < 1000*9; i++)
	{
		Sleep(60*10);
		if(*term == true) break;
	}

	for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_move_inv_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, p2, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
		//error_checking("test9[bit fade test, read]",  i);
		//SHOW_PROGRESS("test9[bit fade test, read]", i, tot_num_blocks);
	}

	return cudaSuccess;
}