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
kernel_test0_read(char* _ptr, char* end_ptr, unsigned int* err, unsigned long* err_addr,
                  unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
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
		RECORD_ERR(err, ptr, pattern, *ptr);
	}

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

		if (*ptr != pattern)
		{
			RECORD_ERR(err, ptr, pattern, *ptr);
		}

		pattern = pattern << 1;
		mask = mask << 1;
	}

	return;
}


int test0(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	unsigned int i;
	char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

	//unsigned long number = 45;
	//cudaMemcpy(err_addr, &number, sizeof(unsigned long), cudaMemcpyHostToDevice);

	//cudaError_t e = cudaGetLastError();
	//return e;
	//test global address
	kernel_test0_global_write<<<1, 1>>>(ptr, end_ptr); SYNC_CUERR;
	kernel_test0_global_read<<<1, 1>>>(ptr, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
	//error_checking("test0 on global address",  0);

	for(int ite = 0; ite < num_iterations; ite++)
	{
		for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
		{
			if(*term == true) break;
			dim3 grid;
			grid.x= GRIDSIZE;
			kernel_test0_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
			//SHOW_PROGRESS("test0 on writing", i, tot_num_blocks);
		}

		for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
		{
			if(*term == true) break;
			dim3 grid;
			grid.x= GRIDSIZE;
			kernel_test0_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
			//error_checking(__FUNCTION__,  i);
			//SHOW_PROGRESS("test0 on reading", i, tot_num_blocks);
		}
	}

	return cudaSuccess;

}