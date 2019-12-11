#include "util.h"
#include <cstdio>

static const int numIterations = 10;

__global__ void relu(float* in, float *out, long long N) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if(p<N) {
		if(in[p]<0)
			out[p] = 0;
		else
			out[p] = in[p];
	}
}

__host__ void execute_relu(float* in, float *out, long long N, const cudaDeviceProp &cdp) {
	dim3 threadsPerBlock(cdp.maxThreadsPerBlock);
	dim3 blocksPerGrid((N+cdp.maxThreadsPerBlock-1)/cdp.maxThreadsPerBlock);
	printf("relu:\n");

	randomize(in, N, -1, 1);
	float *in_dev, *out_dev;
	cudaMalloc(&in_dev, N*sizeof(float));
	cudaMalloc(&out_dev, N*sizeof(float));
	cudaMemcpy(in_dev, in, N*sizeof(float), cudaMemcpyHostToDevice);

	clock_t start, end;
	float totalTime = 0;
	for (int it = 0; it < numIterations; it++) {
		start = clock();
		relu<<<blocksPerGrid, threadsPerBlock>>>(in_dev, out_dev, N);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);

		cudaMemcpy(out, out_dev, N*sizeof(float), cudaMemcpyDeviceToHost);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);

	cudaFree(in_dev);
	cudaFree(out_dev);
}

__host__  void execute_relu_serial(float* in, float *out, long long N) {
	printf("relu_serial:\n");

	randomize(in, N, -1, 1);

	clock_t start, end;
	float totalTime = 0;
	for (int it = 0; it < numIterations; it++) {

		start = clock();
		for(int i=0; i<N; i++) {
			if(in[i]<0)
				out[i] = 0;
			else
				out[i] = in[i];
		}
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}
