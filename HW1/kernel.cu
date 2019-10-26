#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "util.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

extern void printAllCUDASpecs(cudaDeviceProp& deviceProp);
void initPrint();

/*
 * Matrix Addition
 * A = B+C
 */

__global__ void kernel_1t1e(float* A, float* B, float* C, long long N);
__global__ void kernel_1t1r(float* A, float* B, float* C, long long N);
__global__ void kernel_1t1c(float* A, float* B, float* C, long long N);

void execute_1t1e(float *A, float *B, float *C);
void execute_1t1r(float *A, float *B, float *C);
void execute_1t1c(float* A, float* B, float* C);
void execute_serial(float* A, float* B, float* C);
void execute_1t1e_modified(float* A, float* B, float* C);

const long long N = (1<<13);
const int numIterations = 10;
int tpb, tpb_sqrt;

int main()
{
	initPrint();
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("Executing with N=%lld\n\n", N);

	float *A, *B, *C;
	cudaMallocManaged(&A, N * N * sizeof(float));
	cudaMallocManaged(&B, N * N * sizeof(float));
	cudaMallocManaged(&C, N * N * sizeof(float));

	

	tpb = deviceProp.maxThreadsPerBlock;
	tpb_sqrt = sqrt(tpb) + 1e-9;

	execute_1t1e(A, B, C);
	execute_1t1r(A, B, C);
	execute_1t1c(A, B, C);
	execute_serial(A, B, C);
	execute_1t1e_modified(A, B, C);

	cudaFree(A);
	cudaFree(B);
	cudaFree(C);

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

__global__ void kernel_1t1e(float *A, float *B, float *C, long long N) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	if (r < N && c < N) {
		int idx = r * N + c;
		A[idx] = B[idx] + C[idx];
	}
}

__global__ void kernel_1t1r(float* A, float* B, float* C, long long N) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	if (r < N) {
		int idxStart = r * N;
		int idxEnd = idxStart + N;
		int idx;
		for (idx = idxStart; idx < idxEnd; idx++) {
			A[idx] = B[idx] + C[idx];
		}
	}
}

__global__ void kernel_1t1c(float* A, float* B, float* C, long long N) {
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	if (c < N) {
		int idxStart = c;
		int idxEnd = N * N;
		int idx;
		for (idx = idxStart; idx < idxEnd; idx += N) {
			A[idx] = B[idx] + C[idx];
		}
	}
}

__host__ void execute_1t1e(float* A, float* B, float* C) {
	int block_dim = (N + tpb_sqrt - 1) / tpb_sqrt;

	dim3 threadsPerBlock(tpb_sqrt, tpb_sqrt);
	dim3 blocksPerGrid(block_dim, block_dim);
	printf("kernel_1t1e:\tBlocks Per Grid: (%d, %d)\tThreads Per Block: (%d, %d)\n",
		block_dim, block_dim,
		tpb_sqrt, tpb_sqrt);

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(B, N);
		randomize(C, N);

		start = clock();
		kernel_1t1e<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}
__host__ void execute_1t1r(float* A, float* B, float* C) {
	int block_dim = (N + tpb - 1) / tpb;

	dim3 threadsPerBlock(tpb, 1);
	dim3 blocksPerGrid(block_dim, 1);
	printf("kernel_1t1r:\tBlocks Per Grid: (%d, %d)\tThreads Per Block: (%d, %d)\n",
		block_dim, 1,
		tpb, 1);

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(B, N);
		randomize(C, N);

		start = clock();
		kernel_1t1r<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}

__host__  void execute_1t1c(float* A, float* B, float* C) {
	int block_dim = (N + tpb - 1) / tpb;

	dim3 threadsPerBlock(1, tpb);
	dim3 blocksPerGrid(1, block_dim);
	printf("kernel_1t1c:\tBlocks Per Grid: (%d, %d)\tThreads Per Block: (%d, %d)\n",
		1, block_dim,
		1, tpb);

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(B, N);
		randomize(C, N);
		
		start = clock();
		kernel_1t1c<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}

__host__  void execute_serial(float* A, float* B, float* C) {
	printf("serial:\n");

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(B, N);
		randomize(C, N);

		int j,k;
		long long lim = N * N;
		start = clock();
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				int p = j * N + k;
				A[p] = B[p] + C[p];
			}
		}
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}

__host__ void execute_1t1e_modified(float* A, float* B, float* C) {
	int block_dim = (N + tpb - 1) / tpb;

	dim3 threadsPerBlock(1, tpb);
	dim3 blocksPerGrid(N, block_dim);
	printf("kernel_1t1e:\tBlocks Per Grid: (%d, %d)\tThreads Per Block: (%d, %d)\n",
		N, block_dim,
		1, tpb);

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(B, N);
		randomize(C, N);

		start = clock();
		kernel_1t1e<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}