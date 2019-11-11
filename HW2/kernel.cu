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

__global__ void matmul_rec_glob(float* A, float* B, float* C, long long N, long long K, long long M);

void execute_matmul_rec_glob(float *A, float *B, float *C);
void execute_serial(float* A, float* B, float* C);

const long long N = (1 << 8);
const long long K = (1 << 8);
const long long M = (1 << 8);
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
		int p1 = r * M;
		int p2 = c;
		int pos = p1 + c;

		C[pos] = 0;
		for (int i = 0; i < K; i++, p1++, p2+=M) {
			C[pos] += A[p1] * B[p2];
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

__host__  void execute_serial(float* A, float* B, float* C) {
	printf("serial:\n");

	clock_t start, end;
	float totalTime = 0;
	for (int i = 0; i < numIterations; i++) {
		randomize(B, N);
		randomize(C, N);

		int j,k;
		start = clock();

		int p0 = 0;
		for (j = 0; j < N; j++, p0+=M) {
			int pc = p0;//j*M

			for (k = 0; k < M; k++, pc++) {
				int pa = p0;
				int pb = k;
				C[pc] = 0;
				for (int l = 0; l < K; l++, pa++, pb+=M) {
					C[pc] += A[pa] * B[pb];
				}
			}
		}
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}
