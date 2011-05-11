#include "gputests.h"

__global__ void
kernel_test0_global_write(char* _ptr, char* _end_ptr)
{

	unsigned int* ptr = (unsigned int*)_ptr;
	unsigned int* end_ptr = (unsigned int*)_end_ptr;
	unsigned int* orig_ptr = ptr;

	unsigned int pattern = 1;

	unsigned long mask = 4;

	*ptr = pattern;

	while(ptr < end_ptr)
	{

		ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
		if (ptr == orig_ptr)
		{
			mask = mask <<1;
			continue;
		}
		if (ptr >= end_ptr)
		{
			break;
		}

		*ptr = pattern;

		pattern = pattern << 1;
		mask = mask << 1;
	}
	return;
}

__global__ void
kernel_test0_global_read(char* _ptr, char* _end_ptr, unsigned int* err, unsigned long* err_addr,
                         unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
{
	unsigned int* ptr = (unsigned int*)_ptr;
	unsigned int* end_ptr = (unsigned int*)_end_ptr;
	unsigned int* orig_ptr = ptr;

	unsigned int pattern = 1;

	unsigned long mask = 4;

	if (*ptr != pattern)
	{
		RECORD_ERR(err, ptr, pattern, *ptr);
	}

	while(ptr < end_ptr)
	{

		ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
		if (ptr == orig_ptr)
		{
			mask = mask <<1;
			continue;
		}
		if (ptr >= end_ptr)
		{
			break;
		}

		if (*ptr != pattern)
		{
			RECORD_ERR(err, ptr, pattern, *ptr);
		}

		pattern = pattern << 1;
		mask = mask << 1;

		//RECORD_ERR(err, ptr, pattern, *ptr);
	}

	return;
}



__global__ void
kernel_test0_write(char* _ptr, char* end_ptr)
{

	unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
	unsigned int* ptr = orig_ptr;
	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

	unsigned int pattern = 1;

	unsigned long mask = 4;

	*ptr = pattern;

	while(ptr < block_end)
	{

		ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
		if (ptr == orig_ptr)
		{
			mask = mask <<1;
			continue;
		}
		if (ptr >= block_end)
		{
			break;
		}

		*ptr = pattern;

		pattern = pattern << 1;
		mask = mask << 1;
	}
	return;
}


__global__ void
kernel_test0_read(char* _ptr, MemoryError *local_errors, unsigned int *local_count)
{

	unsigned int* orig_ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);;
	unsigned int* ptr = orig_ptr;
	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int* block_end = orig_ptr + BLOCKSIZE/sizeof(unsigned int);

	unsigned int pattern = 1;

	unsigned long mask = 4;
	if (*ptr != pattern)
	{
		RECORD_ERR(local_errors, local_count, ptr, pattern, *ptr);
	}

	while(ptr < block_end)
	{

		ptr = (unsigned int*) ( ((unsigned long)orig_ptr) | mask);
		if (ptr == orig_ptr)
		{
			mask = mask << 1;
			continue;
		}
		if (ptr >= block_end)
		{
			break;
		}

		if (*ptr != pattern)
		{
			RECORD_ERR(local_errors, local_count, ptr, pattern, *ptr);
		}

		pattern = pattern << 1;
		mask = mask << 1;
	}

	return;
}


int test0(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	char* end_ptr = ptr + tot_num_blocks * BLOCKSIZE;
	cudaError_t retval = cudaSuccess;
	MemoryError *local_errors = 0;
	unsigned int *local_count = 0;

	// Allocate temporary memory storage for individual tests, based on GRIDSIZE
	retval = cudaMalloc(&local_errors, GRIDSIZE * sizeof(MemoryError)); if(retval != cudaSuccess) return retval;
	retval = cudaMalloc(&local_count, sizeof(unsigned int)); if(retval != cudaSuccess) return retval;

	kernel_test0_global_write<<<1, 1>>>(ptr, end_ptr); SYNC_CUERR;
	kernel_test0_global_read<<<1, 1>>>(ptr, end_ptr, local_errors, local_count); SYNC_CUERR;

	for(int ite = 0; ite < num_iterations; ite++)
	{
		for (unsigned int i=0; i < tot_num_blocks; i += GRIDSIZE)
		{
			if(*term == true) break;
			dim3 grid;
			grid.x= GRIDSIZE;
			kernel_test0_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
		}

		for (unsigned int i=0; i < tot_num_blocks; i += GRIDSIZE)
		{
			if(*term == true) break;
			dim3 grid;
			grid.x= GRIDSIZE;
			kernel_test0_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
		}
	}

	return retval;
}