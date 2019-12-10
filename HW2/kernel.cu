#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "util.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

extern void printAllCUDASpecs(cudaDeviceProp& deviceProp);
void initPrint();

/*
 * Matrix Multiplication
 * C = AB
 */

#define TILE_WIDTH 32

__global__ void matmul_rec_glob(float* A, float* B, float* C, long long N, long long K, long long M);
__global__ void matmul_rec_shar(float* A, float* B, float* C, long long N, long long K, long long M);

void execute_matmul_rec_glob(float *A, float *B, float *C);
void execute_matmul_rec_shar(float* A, float* B, float* C);
void execute_serial(float* A, float* B, float* C);

const long long N = (1 << 10);
const long long K = (1 << 10);
const long long M = (1 << 10);
const int numIterations = 10;
int tpb, tpb_sqrt;

int main()
{
	initPrint();
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	printf("Executing with N=%lld, K=%lld, M=%lld\n", N, M, K);

	float *A, *B, *C;
	cudaMallocManaged(&A, N * K * sizeof(float));
	cudaMallocManaged(&B, K * M * sizeof(float));
	cudaMallocManaged(&C, N * M * sizeof(float));

	

	tpb = deviceProp.maxThreadsPerBlock;
	tpb_sqrt = sqrt(tpb) + 1e-9;

	execute_matmul_rec_glob(A, B, C);
	execute_matmul_rec_shar(A, B, C);
	execute_serial(A, B, C);

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

__global__ void matmul_rec_glob(float* A, float* B, float* C, long long N, long long K, long long M) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;
	if (r < N && c < M) {
		int pos = r * M + c;
		C[pos] = 0;
		for (int i = 0; i < K; i++) {
			C[pos] += A[r*K+i] * B[i*M+c];
		}
	}
}

__global__ void matmul_rec_shar(float* A, float* B, float* C, long long N, long long K, long long M) {
	int r = blockIdx.x * blockDim.x + threadIdx.x;
	int c = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ float A_tiled[TILE_WIDTH][TILE_WIDTH];
	__shared__ float B_tiled[TILE_WIDTH][TILE_WIDTH];
	if (r < N && c < M) {
		int pos = r * M + c;
		C[pos] = 0;
		for (int i = 0; i < K; i += TILE_WIDTH) {
			//collab loading
			if (i + threadIdx.y < K) {
				A_tiled[threadIdx.x][threadIdx.y] = A[r * K + i + threadIdx.y];
			}
			if (i + threadIdx.x < K) {
				B_tiled[threadIdx.x][threadIdx.y] = B[(i + threadIdx.x) * M + c];
			}
			__syncthreads();

			for (int j = 0; j < TILE_WIDTH && i + j < K; j++) {
				C[pos] += A_tiled[threadIdx.x][j] * B_tiled[j][threadIdx.y];
			}
			__syncthreads();
		}
	}
}

__host__ void execute_matmul_rec_glob(float* A, float* B, float* C) {
	int N_dim = (N + tpb_sqrt - 1) / tpb_sqrt;
	int M_dim = (M + tpb_sqrt - 1) / tpb_sqrt;

	dim3 threadsPerBlock(tpb_sqrt, tpb_sqrt);
	dim3 blocksPerGrid(N_dim, M_dim);
	printf("kernel_matmul_rec_glob:\n");

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(A, N);
		randomize(B, N);

		start = clock();
		matmul_rec_glob<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, K, M);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}

__host__ void execute_matmul_rec_shar(float* A, float* B, float* C) {
	int N_dim = (N + TILE_WIDTH - 1) / TILE_WIDTH;
	int M_dim = (M + TILE_WIDTH - 1) / TILE_WIDTH;

	dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 blocksPerGrid(N_dim, M_dim);
	printf("kernel_matmul_rec_shar:\n");

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(A, N);
		randomize(B, N);

		start = clock();
		matmul_rec_shar<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, K, M);
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
		start = clock();

		for (int i = 0; i < N; i++) {
			for (int j = 0; j < M; j++) {
				int pos = i * M + j;
				C[pos] = 0;
				for (int k = 0; k < K; k++) {
					C[pos] += A[i*K+k] * B[k*M+j];
				}
			}
		}
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}
