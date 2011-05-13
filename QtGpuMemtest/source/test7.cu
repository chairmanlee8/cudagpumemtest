#include "gputests.h"

/******************************************************************************
 * Test 7 [Random number sequence]
 *
 * This test writes a series of random numbers into memory.  A block (1 MB) of memory
 * is initialized with random patterns. These patterns and their complements are
 * used in moving inversions test with rest of memory.
 *
 *
 *******************************************************************************/


__global__ void
kernel_test7_write(char* _ptr, char* end_ptr, char* _start_ptr)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
	unsigned int* start_ptr = (unsigned int*) _start_ptr;

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}


	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		ptr[i] = start_ptr[i];
	}

	return;
}



__global__ void
kernel_test7_readwrite(char* _ptr, char* end_ptr, char* _start_ptr, MemoryError *local_error, int *local_count)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
	unsigned int* start_ptr = (unsigned int*) _start_ptr;

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}


	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		if (ptr[i] != start_ptr[i])
		{
			record_error(local_error, local_count, &ptr[i], start_ptr[i]);
		}
		ptr[i] = ~(start_ptr[i]);
	}

	return;
}

__global__ void
kernel_test7_read(char* _ptr, char* end_ptr, char* _start_ptr, MemoryError *local_error, int *local_count)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);
	unsigned int* start_ptr = (unsigned int*) _start_ptr;

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}


	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		if (ptr[i] != ~(start_ptr[i]))
		{
			record_error(local_error, local_count, &ptr[i], ~(start_ptr[i]));
		}
	}

	return;
}


int
test7(TestInputParams* tip, TestOutputParams *top, bool *term)
{

	unsigned int* host_buf;
	host_buf = (unsigned int*)malloc(BLOCKSIZE);
	unsigned int err = 0;
	unsigned int i;
	unsigned int iteration = 0;

	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		host_buf[i] = get_random_num();
	}

	cudaMemcpy(tip->ptr, host_buf, BLOCKSIZE, cudaMemcpyHostToDevice);


	char* end_ptr = tip->ptr + tip->tot_num_blocks* BLOCKSIZE;

repeat:

	for (i=1; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test7_write<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr, tip->ptr); SYNC_CUERR;
		//SHOW_PROGRESS("test7_write", i, tot_num_blocks);
	}


	for (i=1; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test7_readwrite<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr, tip->ptr, top->err_vector, top->err_count); SYNC_CUERR;
		//err += error_checking("test7_readwrite",  i);
		//SHOW_PROGRESS("test7_readwrite", i, tot_num_blocks);
	}


	for (i=1; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test7_read<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr, tip->ptr, top->err_vector, top->err_count); SYNC_CUERR;
		//err += error_checking("test7_read",  i);
		//SHOW_PROGRESS("test7_read", i, tot_num_blocks);
	}


	if (err == 0 && iteration == 0)
	{
		return cudaSuccess;
	}
	if (iteration < tip->num_iterations)
	{
		//if(*term == true) break;
		//PRINTF("%dth repeating test7 because there are %d errors found in last run\n", iteration, err);
		iteration++;
		err = 0;
		if(*term == false) goto repeat;
	}

	return cudaSuccess;
}