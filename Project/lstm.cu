#include "util.h"
#include <cstdio>
#include <cmath>

static const int numIterations = 10;

__host__ __device__ float sigmoid(float x) {
	return x/(1+exp(-x));
}

__global__ void lstm(float* in, float *out, float *W, float *B, float *U, long long N, long long M) {
	long long p = blockIdx.x * blockDim.x + threadIdx.x;
	long long pos = p;

	float w_f = W[p];
	float w_i = W[p+M];
	float w_o = W[p+M*2];
	float w_c = W[p+M*3];

	float b_f = B[p];
	float b_i = B[p+M];
	float b_o = B[p+M*2];
	float b_c = W[p+M*3];

	float u_f = U[p];
	float u_i = U[p+M];
	float u_o = U[p+M*2];
	float u_c = U[p+M*3];

	float f, i, o, c=0, h=0;
	float x;
	for(int index=0; index<N; index++, pos+=M) {
		x = in[pos];
		f = sigmoid(w_f*x + u_f*h + b_f);
		i = sigmoid(w_i*x + u_i*h + b_i);
		o = sigmoid(w_o*x + u_o*h + b_o);
		c = f*c + i*tanh(w_c*x + u_c*h + b_c);
		h = o*sigmoid(c);
		out[pos] = h;
	}
}

__host__ void execute_lstm(float* in, float *out, long long N, long long M, const cudaDeviceProp &cdp) {
	dim3 threadsPerBlock(cdp.maxThreadsPerBlock);
	dim3 blocksPerGrid((M+cdp.maxThreadsPerBlock-1)/cdp.maxThreadsPerBlock);
	printf("lstm:\n");

	randomize(in, M, -1, 1);
	float *in_dev, *out_dev;
	cudaMalloc(&in_dev, N*M*sizeof(float));
	cudaMalloc(&out_dev, N*M*sizeof(float));
	cudaMemcpy(in_dev, in, N*M*sizeof(float), cudaMemcpyHostToDevice);

	float *W_dev, *B_dev, *U_dev;
	cudaMalloc(&W_dev, M*4*sizeof(float));
	cudaMalloc(&B_dev, M*4*sizeof(float));
	cudaMalloc(&U_dev, M*4*sizeof(float));

	float *W, *B, *U;

	W = (float*)malloc(M*4*sizeof(float));
	randomize(W, M*4, -1, 1);
	cudaMemcpy(W_dev, W, M*4*sizeof(float), cudaMemcpyHostToDevice);

	B = (float*)malloc(M*4*sizeof(float));
	randomize(B, M*4, -1, 1);
	cudaMemcpy(B_dev, B, M*4*sizeof(float), cudaMemcpyHostToDevice);

	U = (float*)malloc(M*4*sizeof(float));
	randomize(U, M*4, -1, 1);
	cudaMemcpy(U_dev, U, M*4*sizeof(float), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	free(W);
	free(B);
	free(U);

	clock_t start, end;
	float totalTime = 0;
	for (int it = 0; it < numIterations; it++) {
		start = clock();
		lstm<<<blocksPerGrid, threadsPerBlock>>>(in_dev, out_dev, W_dev, B_dev, U_dev, N, M);
		cudaDeviceSynchronize();
		end = clock();

		totalTime += (end - start);

		cudaMemcpy(out, out_dev, N*M*sizeof(float), cudaMemcpyDeviceToHost);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);

	cudaFree(in_dev);
	cudaFree(out_dev);
	cudaFree(W_dev);
	cudaFree(B_dev);
	cudaFree(U_dev);
}

__host__  void execute_lstm_serial(float* in, float *out, long long N, long long M) {
	printf("lstm_serial:\n");

	randomize(in, N, -1, 1);

	float *W, *B, *U;
	W = (float*)malloc(M*4*sizeof(float));
	B = (float*)malloc(M*4*sizeof(float));
	U = (float*)malloc(M*4*sizeof(float));

	randomize(W, M*4, -1, 1);
	randomize(B, M*4, -1, 1);
	randomize(U, M*4, -1, 1);

	clock_t start, end;
	float totalTime = 0;
	for (int it = 0; it < numIterations; it++) {

		start = clock();
		float x,f,i,o,c=0,h=0;
		for(long long j=0; j<M; j++) {
			float w_f = W[j];
			float w_i = W[j+M];
			float w_o = W[j+M*2];
			float w_c = W[j+M*3];

			float b_f = B[j];
			float b_i = B[j+M];
			float b_o = B[j+M*2];
			float b_c = W[j+M*3];

			float u_f = U[j];
			float u_i = U[j+M];
			float u_o = U[j+M*2];
			float u_c = U[j+M*3];

			for(long long k=0; k<N; k++) {
				long long pos = k*M+j;
				x = in[pos];
				f = sigmoid(w_f*x + u_f*h + b_f);
				i = sigmoid(w_i*x + u_i*h + b_i);
				o = sigmoid(w_o*x + u_o*h + b_o);
				c = f*c + i*tanh(w_c*x + u_c*h + b_c);
				h = o*sigmoid(c);
				out[pos] = h;
			}
		}
		end = clock();

		totalTime += (end - start);
	}

	printf("Average time elapsed: %fms\n\n", totalTime * 1000.0 / CLOCKS_PER_SEC / numIterations);
}
