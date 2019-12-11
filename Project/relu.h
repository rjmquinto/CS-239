/*
 * relu.h
 *
 *  Created on: Dec 11, 2019
 *      Author: josh
 */

#ifndef RELU_H_
#define RELU_H_

__global__ void relu(float* in, float *out, long long N);
__host__ void execute_relu(float *in, float *out, long long N, const cudaDeviceProp &cdp);
__host__ void execute_relu_serial(float* in, float *out, long long N);


#endif /* RELU_H_ */
