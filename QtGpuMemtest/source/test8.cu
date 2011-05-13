#include "gputests.h"

/***********************************************************************************
 * Test 8 [Modulo 20, random pattern]
 *
 * A random pattern is generated. This pattern is used to set every 20th memory location
 * in memory. The rest of the memory location is set to the complimemnt of the pattern.
 * Repeat this for 20 times and each time the memory location to set the pattern is shifted right.
 *
 *
 **********************************************************************************/

__global__ void
kernel_modtest_write(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int p2)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	for (i = offset; i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ)
	{
		ptr[i] =p1;
	}

	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		if (i % MOD_SZ != offset)
		{
			ptr[i] =p2;
		}
	}

	return;
}


__global__ void
kernel_modtest_read(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, MemoryError* local_error, int* local_count)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	for (i = offset; i < BLOCKSIZE/sizeof(unsigned int); i+=MOD_SZ)
	{
		if (ptr[i] !=p1)
		{
			record_error(local_error, local_count, &ptr[i], p1);
		}
	}

	return;
}

unsigned int
modtest(char* ptr, unsigned int tot_num_blocks, unsigned int offset, unsigned int p1, unsigned int p2, MemoryError *local_error, int *local_count, bool* term)
{

	unsigned int i;
	char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;
	unsigned int err = 0;

	for (i= 0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_modtest_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, offset, p1, p2); SYNC_CUERR;
		//SHOW_PROGRESS("test8[mod test, write]", i, tot_num_blocks);
	}

	for (i= 0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_modtest_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, offset, p1, local_error, local_count); SYNC_CUERR;
		//err += error_checking("test8[mod test, read", i);
		//SHOW_PROGRESS("test8[mod test, read]", i, tot_num_blocks);
	}

	return err;

}

int
test8(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	unsigned int i;
	unsigned int err = 0;
	unsigned int iteration = 0;

	unsigned int p1;
	//if (global_pattern){
	//p1 = global_pattern;
	//}else{
	p1= get_random_num();
	//}
	unsigned int p2 = ~p1;

repeat:

	//PRINTF("test8[mod test]: p1=0x%x, p2=0x%x\n", p1,p2);
	for (i = 0; i < MOD_SZ; i++)
	{
		err += modtest(tip->ptr, tip->tot_num_blocks,i, p1, p2, top->err_vector, top->err_count, term);
	}

	if (err == 0 && iteration == 0)
	{
		return cudaSuccess;
	}

	if (iteration < tip->num_iterations)
	{
		//PRINTF("%dth repeating test8 because there are %d errors found in last run, p1=%x, p2=%p\n", iteration, err, p1, p2);
		iteration++;
		err = 0;
		if(*term == false) goto repeat;
	}

	return cudaSuccess;
}