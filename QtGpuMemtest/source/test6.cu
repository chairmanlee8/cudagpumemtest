#include "gputests.h"

/*****************************************************************************************
 * Test 6 [Moving inversions, 32 bit pat]
 * This is a variation of the moving inversions algorithm that shifts the data
 * pattern left one bit for each successive address. The starting bit position
 * is shifted left for each pass. To use all possible data patterns 32 passes
 * are required.  This test is quite effective at detecting data sensitive
 * errors but the execution time is long.
 *
 ***************************************************************************************/



__global__ void
kernel_movinv32_write(char* _ptr, char* end_ptr, unsigned int pattern,
                      unsigned int lb, unsigned int sval, unsigned int offset)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int k = offset;
	unsigned pat = pattern;
	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		ptr[i] = pat;
		k++;
		if (k >= 32)
		{
			k=0;
			pat = lb;
		}
		else
		{
			pat = pat << 1;
			pat |= sval;
		}
	}

	return;
}


__global__ void
kernel_movinv32_readwrite(char* _ptr, char* end_ptr, unsigned int pattern,
                          unsigned int lb, unsigned int sval, unsigned int offset, MemoryError* local_error, int* local_count)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int k = offset;
	unsigned pat = pattern;
	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		if (ptr[i] != pat)
		{
			record_error(local_error, local_count, &ptr[i], pat);
		}

		ptr[i] = ~pat;

		k++;
		if (k >= 32)
		{
			k=0;
			pat = lb;
		}
		else
		{
			pat = pat << 1;
			pat |= sval;
		}
	}

	return;
}



__global__ void
kernel_movinv32_read(char* _ptr, char* end_ptr, unsigned int pattern,
                     unsigned int lb, unsigned int sval, unsigned int offset, MemoryError *local_error, int *local_count)
{
	unsigned int i;
	unsigned int* ptr = (unsigned int*) (_ptr + blockIdx.x*BLOCKSIZE);

	if (ptr >= (unsigned int*) end_ptr)
	{
		return;
	}

	unsigned int k = offset;
	unsigned pat = pattern;
	for (i = 0; i < BLOCKSIZE/sizeof(unsigned int); i++)
	{
		if (ptr[i] != ~pat)
		{
			record_error(local_error, local_count, &ptr[i], ~pat);
		}

		k++;
		if (k >= 32)
		{
			k=0;
			pat = lb;
		}
		else
		{
			pat = pat << 1;
			pat |= sval;
		}
	}

	return;
}



int
movinv32(char* ptr, unsigned int tot_num_blocks, unsigned int pattern,
         unsigned int lb, unsigned int sval, unsigned int offset, MemoryError* local_error, int *local_count, bool *term)
{

	unsigned int i;

	char* end_ptr = ptr + tot_num_blocks* BLOCKSIZE;

	for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_movinv32_write<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset); SYNC_CUERR;
		//SHOW_PROGRESS("test6[moving inversion 32 write]", i, tot_num_blocks);
	}

	for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_movinv32_readwrite<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset, local_error, local_count); SYNC_CUERR;
		//error_checking("test6[moving inversion 32 readwrite]",  i);
		//SHOW_PROGRESS("test6[moving inversion 32 readwrite]", i, tot_num_blocks);
	}

	for (i=0; i < tot_num_blocks; i+= GRIDSIZE)
	{
		if(*term == true) break;
		dim3 grid;
		grid.x= GRIDSIZE;
		kernel_movinv32_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, pattern, lb,sval, offset, local_error, local_count); SYNC_CUERR;
		//error_checking("test6[moving inversion 32 read]",  i);
		//SHOW_PROGRESS("test6[moving inversion 32 read]", i, tot_num_blocks);
	}

	return cudaSuccess;

}


int
test6(TestInputParams *tip, TestOutputParams *top, bool *term)
{
	unsigned int i;

	unsigned int pattern;

	for (i= 0, pattern = 1; i < 32; pattern = pattern << 1, i++)
	{

		//DEBUG_PRINTF("Test6[move inversion 32 bits test]: pattern =0x%x, offset=%d\n", pattern, i);
		movinv32(tip->ptr, tip->tot_num_blocks, pattern, 1, 0, i, top->err_vector, top->err_count, term);
		//DEBUG_PRINTF("Test6[move inversion 32 bits test]: pattern =0x%x, offset=%d\n", ~pattern, i);
		movinv32(tip->ptr, tip->tot_num_blocks, ~pattern, 0xfffffffe, 1, i, top->err_vector, top->err_count, term);

	}
	return cudaSuccess;

}
