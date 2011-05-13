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

	// Allocate small memory
	cudaMalloc((void**)&detectedErrors, MAX_ERR_RECORD_COUNT * sizeof(MemoryError));
	CUERR("Could not allocate memory for errors.");

	cudaMalloc((void**)&numberErrors, sizeof(int));
	CUERR("Could not allocate memory for number of errors.");

	char* ptr = NULL;

	tot_num_blocks += 1;
	do
	{
		tot_num_blocks--;

		if(tot_num_blocks <= 0)
		{
			emit log(TestInfo(), QString("Could not allocate any memory on this device."));
			return;
		}

		cudaMalloc((void**) &ptr, tot_num_blocks * BLOCKSIZE);
	}
	while(cudaGetLastError() != cudaSuccess);

	// Run tests
	do { run_tests_impl(ptr, tot_num_blocks); } while (infiniteFlag & !terminationFlag);

	cudaFree(ptr);
}

void QtGpuThread::run_tests_impl(char* ptr, unsigned int tot_num_blocks)
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
					emit testEnded(tests[j]);
			}
			break;
		}
		if(!tests[i].testEnabled) continue;

		//emit log(device, tests[i].testName, QString("Test started."));
		emit testStarting(tests[i]);

		TestInputParams * tip = new TestInputParams;
		TestOutputParams * top = new TestOutputParams;

		tip->ptr = ptr;
		tip->tot_num_blocks = tot_num_blocks;
		tip->num_iterations = 1;
		top->err_vector = detectedErrors;
		top->err_count = numberErrors;

		int returnCode = (*(tests[i].testFunc))(tip, top, &terminationFlag);

		if(returnCode != cudaSuccess)
		{
			QString errString;
			QTextStream(&errString) << "CUDA failure (err code: " << returnCode << ") occurred while running test.";

			emit log(tests[i], errString);
			emit testFailed(tests[i]);
		}
		else
		{
			int local_count = 0;
			MemoryError *local_errors = new MemoryError[MAX_ERR_RECORD_COUNT];

			cudaMemcpy(&local_count, top->err_count, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(local_errors, top->err_vector, sizeof(MemoryError) * MAX_ERR_RECORD_COUNT, cudaMemcpyDeviceToHost);

			// log the result
			if(local_count > 0)
			{
				// got some memory errors, record them and emit fail
				for(int n = 0; n < local_count; n++)
				{
					// log each error
					QString memError;
					QTextStream memErrorStream(&memError);
					memErrorStream << qSetFieldWidth(8) << qSetPadChar('0') << right << hex << "Error: Address 0x" << local_errors[n].addr << " has 0x" << local_errors[n].current << " but expected 0x" << local_errors[n].expected;
					memErrorStream.flush();

					emit log(tests[i], memError);
				}

				emit testFailed(tests[i]);
			}
			else
			{
				emit testPassed(tests[i]);
			}

			delete [] local_errors;
		}

		//TODO: I commented the emit's here out because stress test will produce too many of these
		//emit log(device, tests[i].testName, QString("Test ended."));
		emit testEnded(tests[i]);
	}
}

// this function is for general errors that aren't bound to any specific test
cudaError_t QtGpuThread::cudaError(QString msgFail)
{
	cudaError_t cuda_err;
	if((cuda_err = cudaGetLastError()) != cudaSuccess)
	{
		emit log(TestInfo(), msgFail);
	}

	return cuda_err;
}