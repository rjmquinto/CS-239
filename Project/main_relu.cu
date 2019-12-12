#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "util.h"
#include "relu.h"
#include <stdio.h>
#include <time.h>

void checkError() {
    cudaError_t err = cudaGetLastError();
    if(err) {
        printf("Error: %s\n",cudaGetErrorString(err));
        exit(1);
    }
}

int numIterations = 1000;

int main()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

    float *in, *out;
	long long N = readData(&in, "relu/in.dat");
	out = (float*)malloc(N * sizeof(float));

    // printf("Executing with N=%lld\n", N);

	dim3 threadsPerBlock(deviceProp.maxThreadsPerBlock);
	dim3 blocksPerGrid((N+deviceProp.maxThreadsPerBlock-1)/deviceProp.maxThreadsPerBlock);
	// printf("relu:\n");

	float *in_dev, *out_dev;
	do {
		cudaMalloc(&in_dev, N*sizeof(float));
    } while(cudaGetLastError());

	do {
		cudaMalloc(&out_dev, N*sizeof(float));
	} while(cudaGetLastError());

	do {
		cudaMemcpy(in_dev, in, N*sizeof(float), cudaMemcpyHostToDevice);
	} while(cudaGetLastError());

	clock_t start, end;
	float totalTime = 0;

	for(int i=0; i<numIterations; i++)
    {
		start = clock();
		do {
			relu<<<blocksPerGrid, threadsPerBlock>>>(in_dev, out_dev, N);
		} while(cudaGetLastError());
		do {
			cudaDeviceSynchronize();
		} while(cudaGetLastError());
		end = clock();

		totalTime += (end - start);
		printf("%f\n",(end-start)*1000.00/CLOCKS_PER_SEC);

		// cudaMemcpy(out, out_dev, N*sizeof(float), cudaMemcpyDeviceToHost);
	}

	// printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);

	do {
		cudaFree(in_dev);
	} while(cudaGetLastError());
	do {
		cudaFree(out_dev);
	} while(cudaGetLastError());

	free(in);
	free(out);

    return 0;
}
