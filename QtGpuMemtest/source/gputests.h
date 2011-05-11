#ifndef __GPU_TESTS_H
#define __GPU_TESTS_H

//
// Shared defines
//

#include <vector>
#include <algorithm>

#define BLOCKSIZE ((unsigned long)(1024 * 1024))
#define GRIDSIZE 128
#define MOD_SZ 20

struct TestInputParams
{
	char*			ptr;
	unsigned int	tot_num_blocks;
	int				num_iterations;
};

struct TestOutputParams
{
	std::vector<MemoryError>	err_vector;
};

struct MemoryError
{
	unsigned long	start_addr;
	unsigned long	expected;
	unsigned long	current;
	unsigned long	second_read;
	unsigned long	end_addr;
};

#ifndef __CUDACC__

//
// Qt/MSVC defines only
//

#include <QtGui>
#include <cuda_runtime_api.h>

#include "testiconwidget.h"

#define ERROR_NO_MEMORY			1
#define PROGRESS_DONE			1

#define CUERR(msg) do { if(cudaError(QString(msg)) != cudaSuccess) return; } while(0);

class QtGpuMemtest;
struct MemoryError;

typedef int (*TestFunc)(TestInputParams*, TestOutputParams*, bool*);

extern int test0(TestInputParams*, TestOutputParams*, bool*);
/*extern int test1(TestInputParams*, TestOutputParams*, bool*);
extern int test2(TestInputParams*, TestOutputParams*, bool*);
extern int test3(TestInputParams*, TestOutputParams*, bool*);
extern int test4(TestInputParams*, TestOutputParams*, bool*);
extern int test5(TestInputParams*, TestOutputParams*, bool*);
extern int test6(TestInputParams*, TestOutputParams*, bool*);
extern int test7(TestInputParams*, TestOutputParams*, bool*);
extern int test8(TestInputParams*, TestOutputParams*, bool*);
extern int test9(TestInputParams*, TestOutputParams*, bool*);
extern int test10(TestInputParams*, TestOutputParams*, bool*);*/

struct TestInfo
{
	int			testId;
	QString		testName;
	QString		testShortName;		// short name, no more than 2 characters long, for identifying the test in the TestIconWidget.
	TestFunc	testFunc;
	bool		testEnabled;

	TestInfo() : testId(0), testName(QString("")), testShortName(QString("")), testFunc(0), testEnabled(false) { };
	TestInfo(int id, QString name, QString shortName, TestFunc func, bool e) : testId(id), testName(name), testShortName(shortName), testFunc(func), testEnabled(e) {};
};


class QtGpuThread : public QThread
{
	Q_OBJECT

public:
	QtGpuThread(QVector<TestInfo>& aTests, QObject* parent = 0);
	QtGpuThread(QObject* parent = 0) : QThread(parent) { };

	void run();

protected:
	void run_tests(char* ptr, unsigned int tot_num_blocks);

public slots:
	void setDevice(int idx)	{ device = idx;	}

	void notifyExit() { terminationFlag = true; }
	void setEndless(bool b)	{ infiniteFlag = b;	}

signals:
	// Is deviceIdx necessary here? Depends on how far up the chain the signal is connected.
	// Best to err on the safer side and keep it.

	void failed(int deviceIdx, QString testName);
	void passed(int deviceIdx, QString testName);
	void ended(int deviceIdx, QString testName);
	void starting(int deviceIdx, QString testName);
	void log(int deviceIdx, QString testName, QString logMessage);

protected:
	cudaError_t cudaError(QString msgFail);

private:
	unsigned int		device;
	QVector<TestInfo>	tests;
	QList<MemoryError>	detectedErrors;

	bool	terminationFlag;
	bool	infiniteFlag;
};

#else

//
// CUDA tests/kernel defines only
//

#include <windows.h>

#ifdef SM_10
#define atomicAdd(x, y) do{ (*x) = (*x) + y ;}while(0)
#define RECORD_ERR(err, p, expect, current) do{	  \
	atomicAdd(err, 1); \
	}while(0)
#else

/*#define RECORD_ERR(err, p, expect, current) do{		\
	unsigned int idx = atomicAdd(err, 1);		\
	idx = idx % MAX_ERR_RECORD_COUNT;		\
	err_addr[idx] = (unsigned long)p;		\
	err_expect[idx] = (unsigned long)expect;	\
	err_current[idx] = (unsigned long)current;	\
	err_second_read[idx] = (unsigned long)(*p);	\
}while(0)*/

#define RECORD_ERR(err_store, err_count, p, expect, current) do {	\
	unsigned int idx = atomicAdd(err_count, 1);						\
	MemoryError *err_store_ptr = &err_store[idx];					\
	err_store_ptr->addr = (unsigned long) p;						\
	err_store_ptr->expected = (unsigned long) expect;				\
	err_store_ptr->current = (unsigned long) current;				\
	err_store_ptr->second_read = (unsigned long) (*p);				\
} while(0)

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

// Common Functions
unsigned int get_random_num(void);
unsigned long get_random_num_long(void);
/*__global__ void kernel_move_inv_write(char* _ptr, char* end_ptr, unsigned int pattern);
__global__ void kernel_move_inv_readwrite(char* _ptr, char* end_ptr, unsigned int p1, unsigned int p2, unsigned int* err, unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read);
__global__ void kernel_move_inv_read(char* _ptr, char* end_ptr,  unsigned int pattern, unsigned int* err, unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read );*/
unsigned int move_inv_test(char* ptr, unsigned int tot_num_blocks, unsigned int p1, unsigned p2, unsigned int* err_count, unsigned long* err_addr, unsigned long* err_expect, unsigned long* err_current, unsigned long* err_second_read, bool *term);

#endif

#endif

#endif