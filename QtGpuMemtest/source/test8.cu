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

extern unsigned int
get_random_num(void);



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
kernel_modtest_read(char* _ptr, char* end_ptr, unsigned int offset, unsigned int p1, unsigned int* err,
                    unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read)
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
			RECORD_ERR(err, &ptr[i], p1, ptr[i]);
		}
	}

	return;
}

unsigned int
modtest(char* ptr, unsigned int tot_num_blocks, unsigned int offset, unsigned int p1, unsigned int p2, unsigned int* err_count, unsigned long* err_addr,
        unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read, bool* term)
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
		kernel_modtest_read<<<grid, 1>>>(ptr + i*BLOCKSIZE, end_ptr, offset, p1, err_count, err_addr, err_expect, err_current, err_second_read); SYNC_CUERR;
		//err += error_checking("test8[mod test, read", i);
		//SHOW_PROGRESS("test8[mod test, read]", i, tot_num_blocks);
	}

	return err;

}

int
test8(char* ptr, unsigned int tot_num_blocks, int num_iterations, unsigned int* err_count, unsigned long* err_addr,
      unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read, bool *term)
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
		err += modtest(ptr, tot_num_blocks,i, p1, p2, err_count, err_addr, err_expect, err_current, err_second_read, term);
	}

	if (err == 0 && iteration == 0)
	{
		return cudaSuccess;
	}

	if (iteration < MAX_ITERATION)
	{
		//PRINTF("%dth repeating test8 because there are %d errors found in last run, p1=%x, p2=%p\n", iteration, err, p1, p2);
		iteration++;
		err = 0;
		if(*term == false) goto repeat;
	}

	return cudaSuccess;
}