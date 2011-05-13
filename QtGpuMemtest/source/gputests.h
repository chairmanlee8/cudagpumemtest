#ifndef __GPU_TESTS_H
#define __GPU_TESTS_H

//
// Shared defines
//

#include "common.h"

#define BLOCKSIZE ((unsigned long)(1024 * 1024))
#define GRIDSIZE 128
#define MOD_SZ 20
#define MAX_ERR_RECORD_COUNT 16

#ifndef __CUDACC__

//
// Qt/MSVC defines only
//

#include <QtGui>
#include <cuda_runtime_api.h>

#define CUERR(msg) do { if(cudaError(QString(msg)) != cudaSuccess) return; } while(0);

class QtGpuMemtest;

extern int test0(TestInputParams*, TestOutputParams*, bool*);
extern int test1(TestInputParams*, TestOutputParams*, bool*);
extern int test2(TestInputParams*, TestOutputParams*, bool*);
extern int test3(TestInputParams*, TestOutputParams*, bool*);
extern int test4(TestInputParams*, TestOutputParams*, bool*);
extern int test5(TestInputParams*, TestOutputParams*, bool*);
extern int test6(TestInputParams*, TestOutputParams*, bool*);
extern int test7(TestInputParams*, TestOutputParams*, bool*);
extern int test8(TestInputParams*, TestOutputParams*, bool*);
extern int test9(TestInputParams*, TestOutputParams*, bool*);
extern int test10(TestInputParams*, TestOutputParams*, bool*);

class QtGpuThread : public QThread
{
	Q_OBJECT

protected:
	void run();
	void run_tests_impl(char* ptr, unsigned int tot_num_blocks);

public:
	QtGpuThread(QVector<TestInfo>& aTests, QObject* parent = 0);
	QtGpuThread(QObject* parent = 0) : QThread(parent) { };

	void setDevice(int idx)	{ device = idx;	}
	int deviceIndex() { return device; }

	int totalProgressParts();

public slots:
	void notifyExit() { terminationFlag = true; }
	void setEndless(bool b)	{ infiniteFlag = b;	}

signals:
	void testFailed(TestInfo test);
	void testPassed(TestInfo test);
	void testEnded(TestInfo test);
	void testStarting(TestInfo test);
	void progressPart();
	void log(TestInfo test, QString logMessage);

protected:
	cudaError_t cudaError(QString msgFail);

private:
	unsigned int		device;
	QVector<TestInfo>	tests;

	MemoryError			*detectedErrors;
	int					*numberErrors;

	bool				terminationFlag;
	bool				infiniteFlag;
};

#else

//
// CUDA tests/kernel defines only
//

#include <windows.h>

#define SYNC_CUERR do { cudaError_t __err; cudaThreadSynchronize(); \
	if((__err = cudaGetLastError()) != cudaSuccess) return __err; \
} while(0)

#define CUERR do { cudaError_t __err; \
	if((__err = cudaGetLastError()) != cudaSuccess) return __err; \
} while(0)

#define SHOW_PROGRESS
#define DEBUG_PRINTF

#define STRESS_BLOCKSIZE 64
#define STRESS_GRIDSIZE (1024*32)

__forceinline__ __device__ void record_error(MemoryError *local_errors, int *local_count, unsigned long *ptr, unsigned long pattern)
{
	unsigned int idx = atomicAdd(local_count, 1);
	idx = idx % MAX_ERR_RECORD_COUNT;
	local_errors[idx].addr = (unsigned long) ptr;
	local_errors[idx].expected = (unsigned long) pattern;
	local_errors[idx].current = (unsigned long) *ptr;
	local_errors[idx].second_read = (unsigned long) (*ptr);
}

__forceinline__ __device__ void record_error(MemoryError *local_errors, int *local_count, unsigned int *ptr, unsigned int pattern)
{
	unsigned int idx = atomicAdd(local_count, 1);
	idx = idx % MAX_ERR_RECORD_COUNT;
	local_errors[idx].addr = (unsigned long) ptr;
	local_errors[idx].expected = (unsigned long) pattern;
	local_errors[idx].current = (unsigned long) *ptr;
	local_errors[idx].second_read = (unsigned long) (*ptr);
}

// Common Functions
unsigned int get_random_num(void);
unsigned long get_random_num_long(void);
__global__ void kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern);
__global__ void kernel_move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, MemoryError* local_errors, int* local_count);
__global__ void kernel_move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, MemoryError* local_errors, int* local_count);
unsigned int move_inv_test(char* ptr, unsigned int tot_num_blocks, unsigned int p1, unsigned p2, MemoryError* local_errors, int* local_count, bool *term);

#endif

#endif