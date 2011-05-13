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
kernel_test0_global_read(char* _ptr, char* _end_ptr, MemoryError *local_errors, int *local_count)
{
	unsigned int* ptr = (unsigned int*)_ptr;
	unsigned int* end_ptr = (unsigned int*)_end_ptr;
	unsigned int* orig_ptr = ptr;

	unsigned int pattern = 1;

	unsigned long mask = 4;

	if (*ptr != pattern)
	{
		record_error(local_errors, local_count, ptr, pattern);
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
			record_error(local_errors, local_count, ptr, pattern);
		}

		pattern = pattern << 1;
		mask = mask << 1;

		//record_error(local_errors, local_count, ptr, pattern);
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
kernel_test0_read(char* _ptr, char* end_ptr, MemoryError *local_errors, int *local_count)
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
		record_error(local_errors, local_count, ptr, pattern);
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
			record_error(local_errors, local_count, ptr, pattern);
		}

		pattern = pattern << 1;
		mask = mask << 1;
	}

	return;
}


int test0(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	char* end_ptr = tip->ptr + tip->tot_num_blocks * BLOCKSIZE;

	kernel_test0_global_write<<<1, 1>>>(tip->ptr, end_ptr); SYNC_CUERR;
	kernel_test0_global_read<<<1, 1>>>(tip->ptr, end_ptr, top->err_vector, top->err_count); SYNC_CUERR;

	for(int ite = 0; ite < tip->num_iterations; ite++)
	{
		for (unsigned int i=0; i < tip->tot_num_blocks; i += GRIDSIZE)
		{
			if(*term == true) break;
			dim3 grid;
			grid.x= GRIDSIZE;
			kernel_test0_write<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
		}

		for (unsigned int i=0; i < tip->tot_num_blocks; i += GRIDSIZE)
		{
			if(*term == true) break;
			dim3 grid;
			grid.x= GRIDSIZE;
			kernel_test0_read<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr, top->err_vector, top->err_count); SYNC_CUERR;
		}
	}

	return cudaSuccess;
}