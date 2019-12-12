#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "util.h"
#include "lstm.h"
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

    float *in, *out, *W, *B, *U;
    long long M = readData(&W,"lstm/w.dat");
    M >>= 2;
    readData(&B,"lstm/b.dat");
    readData(&U,"lstm/u.dat");
	long long N = readData(&in, "lstm/in.dat") / M;
	out = (float*)malloc(N * M * sizeof(float));

    // printf("Executing with N=%lld, M=%lld\n", N, M);

	dim3 threadsPerBlock(deviceProp.maxThreadsPerBlock);
	dim3 blocksPerGrid((M+deviceProp.maxThreadsPerBlock-1)/deviceProp.maxThreadsPerBlock);
	// printf("lstm:\n");

	float *in_dev, *out_dev, *W_dev, *B_dev, *U_dev;
    do {
	cudaMalloc(&in_dev, N*M*sizeof(float));
    } while(cudaGetLastError());

    do {
	    cudaMalloc(&out_dev, N*M*sizeof(float));
    } while(cudaGetLastError());
    do {
        cudaMalloc(&W_dev, M*4*sizeof(float));
    } while(cudaGetLastError());
    do {
        cudaMalloc(&B_dev, M*4*sizeof(float));
    } while(cudaGetLastError());
    do {
        cudaMalloc(&U_dev, M*4*sizeof(float));
    } while(cudaGetLastError());

    do {
	    cudaMemcpy(in_dev, in, N*M*sizeof(float), cudaMemcpyHostToDevice);
    } while(cudaGetLastError());
    do {
        cudaMemcpy(W_dev, W, M*4*sizeof(float), cudaMemcpyHostToDevice);
    } while(cudaGetLastError());
    do {
        cudaMemcpy(B_dev, B, M*4*sizeof(float), cudaMemcpyHostToDevice);
    } while(cudaGetLastError());
    do {
        cudaMemcpy(U_dev, U, M*4*sizeof(float), cudaMemcpyHostToDevice);
    } while(cudaGetLastError());

	clock_t start, end;
	float totalTime = 0;

    for(int i=0; i<numIterations; i++)
    {
		start = clock();
        do {
		    lstm<<<blocksPerGrid, threadsPerBlock>>>(in_dev, out_dev, W_dev, B_dev, U_dev, N, M);
        } while(cudaGetLastError());
        do {
		    cudaDeviceSynchronize();
        } while(cudaGetLastError());
		end = clock();

        checkError();

		totalTime += (end - start);
        printf("%f\n",(end-start)*1000.00/CLOCKS_PER_SEC);

		// cudaMemcpy(out, out_dev, N*M*sizeof(float), cudaMemcpyDeviceToHost);
        checkError();
	}

	// printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);

    do {
	    cudaFree(in_dev);
    } while(cudaGetLastError());
    do {
	    cudaFree(out_dev);
    } while(cudaGetLastError());
    do {
        cudaFree(W_dev);
    } while(cudaGetLastError());
    do {
        cudaFree(B_dev);
    } while(cudaGetLastError());
    do {
        cudaFree(U_dev);
    } while(cudaGetLastError());

	free(in);
    free(W);
    free(B);
    free(U);
	free(out);

    return 0;
}
