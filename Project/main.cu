#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "util.h"
#include "relu.h"
#include "lstm.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

extern void printAllCUDASpecs(cudaDeviceProp& deviceProp);
void initPrint();

/*
 * Matrix Multiplication
 * C = AB
 */

void runReluExperiment(cudaDeviceProp &deviceProp)
{
	const long long N = (1ll << 25);
	printf("Executing with N=%lld\n", N);

	float *A, *B;
	A = (float*)malloc(N * sizeof(float));
	B = (float*)malloc(N * sizeof(float));

	execute_relu(A, B, N, deviceProp);
	execute_relu_serial(A, B, N);

	free(A);
	free(B);
}

void runLSTMExperiment(cudaDeviceProp &deviceProp)
{
	const long long N = (1ll << 10);
	const long long M = (1ll << 15);
	printf("Executing with N=%lld, M=%lld\n", N, M);

	float *A, *B;
	A = (float*)malloc(N*M*sizeof(float));
	B = (float*)malloc(N*M*sizeof(float));

	execute_lstm(A, B, N, M, deviceProp);
	execute_lstm_serial(A, B, N, M);

	free(A);
	free(B);
}

int main()
{
	initPrint();
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	runReluExperiment(deviceProp);
	runLSTMExperiment(deviceProp);

    return 0;
}

void initPrint() {
	int device;
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	for (device = 0; device < deviceCount; ++device) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("\nDevice %d: \"%s\"\n", device, deviceProp.name);
		printAllCUDASpecs(deviceProp);
	}
}
