#include "Main.h"

__declspec(dllexport) int CUDA_GetNumDevices()
{
	int x;
	cudaGetDeviceCount(&x);

	return x;
}

__declspec(dllexport) int CUDA_GetDeviceProperties()
{
}