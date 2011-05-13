#include "gputests.h"

__global__ void
kernel_test1_write(char* _ptr, char* end_ptr)
{
	unsigned int i;
	unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned long*) end_ptr)
	{
		return;
	}


	for (i = 0; i < BLOCKSIZE/sizeof(unsigned long); i++)
	{
		ptr[i] =(unsigned long) & ptr[i];
	}

	return;
}

__global__ void
kernel_test1_read(char* _ptr, char* end_ptr, MemoryError *local_errors, int *local_count)
{
	unsigned int i;
	unsigned long* ptr = (unsigned long*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned long*) end_ptr)
	{
		return;
	}


	for (i = 0; i < BLOCKSIZE/sizeof(unsigned long); i++)
	{
		if (ptr[i] != (unsigned long)& ptr[i])
		{
			record_error(local_errors, local_count, &ptr[i], (unsigned long)&ptr[i]);
		}
	}

	return;
}



int
test1(TestInputParams *tip, TestOutputParams *top, bool *term)
{


	unsigned int i;
	char* end_ptr = tip->ptr + tip->tot_num_blocks* BLOCKSIZE;

	for (i=0; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test1_write<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr); SYNC_CUERR;
		//SHOW_PROGRESS("test1 on writing", i, tot_num_blocks);

	}

	for (i=0; i < tip->tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_test1_read<<<grid, 1>>>(tip->ptr + i*BLOCKSIZE, end_ptr, top->err_vector, top->err_count); SYNC_CUERR;
		//error_checking("test1 on reading",  i);
		//SHOW_PROGRESS("test1 on reading", i, tot_num_blocks);

	}


	return cudaSuccess;

}