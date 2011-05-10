#include "gputests.h"

QtGpuThread::QtGpuThread(QVector<TestInfo>& aTests, QObject* parent)
	: QThread(), tests(aTests), terminationFlag(false), infiniteFlag(false)
{
}

// equivalent to thread_func in the old code
void QtGpuThread::run()
{
	struct cudaDeviceProp prop;
	cudaError_t err;

	cudaGetDeviceProperties(&prop, device);
	CUERR("Could not get device properties.");

	unsigned long totmem = prop.totalGlobalMem;
	unsigned int tot_num_blocks = totmem / BLOCKSIZE;

	cudaSetDevice(device);
	cudaThreadSynchronize();
	CUERR("Could not set CUDA device as active.");

	char* ptr = NULL;

	tot_num_blocks += 1;
	do
	{
		tot_num_blocks--;

		if(tot_num_blocks <= 0)
		{
			emit log(device, QString(""), QString("Could not allocate any memory on this device."));
			return;
		}

		cudaMalloc((void**) &ptr, tot_num_blocks * BLOCKSIZE);
	}
	while(cudaGetLastError() != cudaSuccess);

	if(infiniteFlag)
	{
		while(!terminationFlag) run_tests(ptr, tot_num_blocks);
	}
	else
	{
		run_tests(ptr, tot_num_blocks);
	}

	cudaFree(ptr);
}

void QtGpuThread::run_tests(char* ptr, unsigned int tot_num_blocks)
{
	// Run each test and do some other stuff as described in the flowchart
	for(int i = 0; i < tests.size(); i++)
	{
		if(terminationFlag)
		{
			// Emit all the test ended flags so progress updates correctly
			for(int j = i; j < tests.size(); j++)
			{
				if(tests[j].testEnabled)
					emit ended(device, tests[j].testName);
			}
			break;
		}
		if(!tests[i].testEnabled) continue;

		//emit log(device, tests[i].testName, QString("Test started."));
		emit starting(device, tests[i].testName);

		TestInputParams * tip = new TestInputParams;
		TestOutputParams * top = new TestOutputParams;

		tip->ptr = ptr;
		tip->tot_num_blocks = tot_num_blocks;
		tip->num_iterations = 1;

		int returnCode = (*(tests[i].testFunc))(tip, top, &terminationFlag);

		if(returnCode != cudaSuccess)
		{
			QString errString;
			QTextStream(&errString) << "CUDA failure (err code: " << returnCode << ") occurred while running test.";

			emit log(device, tests[i].testName, errString);
			emit failed(device, tests[i].testName);
		}
		else
		{
			// see if memory came back correct
			unsigned int* local_err_count = new unsigned int;
			unsigned long* local_err_addr = new unsigned long[MAX_ERR_RECORD_COUNT];
			unsigned long* local_err_expect = new unsigned long[MAX_ERR_RECORD_COUNT];
			unsigned long* local_err_current = new unsigned long[MAX_ERR_RECORD_COUNT];
			unsigned long* local_err_second_read = new unsigned long[MAX_ERR_RECORD_COUNT];

			cudaMemcpy(local_err_count, err_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			cudaMemcpy(local_err_addr, err_addr, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);
			cudaMemcpy(local_err_expect, err_expect, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);
			cudaMemcpy(local_err_current, err_current, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);
			cudaMemcpy(local_err_second_read, err_second_read, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);

			// log the result
			if(*local_err_count != 0)
			{
				// got some memory errors, record them and emit fail
				for(int n = 0; n < std::min<unsigned long>(*local_err_count, MAX_ERR_RECORD_COUNT); n++)
				{
					// log each error
					QString memError;
					QTextStream memErrorStream(&memError);
					memErrorStream << qSetFieldWidth(8) << qSetPadChar('0') << right << hex <<
					               "Error: Address 0x" << local_err_addr[n] << " has 0x" << local_err_current[n] <<
					               " but expected 0x" << local_err_expect[n];
					memErrorStream.flush();
					emit log(device, tests[i].testName, memError);
				}

				emit failed(device, tests[i].testName);
			}
			else
			{
				emit passed(device, tests[i].testName);
			}

			delete local_err_count;
			delete [] local_err_second_read;
			delete [] local_err_current;
			delete [] local_err_expect;
			delete [] local_err_addr;
		}

		//TODO: I commented the emit's here out because stress test will produce too many of these
		//emit log(device, tests[i].testName, QString("Test ended."));
		emit ended(device, tests[i].testName);
	}
}

// this function is for general errors that aren't bound to any specific test
cudaError_t QtGpuThread::cudaError(QString msgFail)
{
	cudaError_t cuda_err;
	if((cuda_err = cudaGetLastError()) != cudaSuccess)
	{
		emit log(device, QString(""), msgFail);
	}

	return cuda_err;
}

/*inline cudaError_t QtGpuThread::cudaError(QString line, QString file)
{
	cudaError_t cuda_err;
	if((cuda_err = cudaGetLastError()) != cudaSuccess)
	{
		emit blockingError(device, 0, cuda_err, line, file);
	}

	return cuda_err;
}

inline cudaError_t QtGpuThread::syncCudaError(QString line, QString file)
{
	cudaThreadSynchronize();
	return cudaError(line, file);
}*/