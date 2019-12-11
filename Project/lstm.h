/*
 * lstm.h
 *
 *  Created on: Dec 11, 2019
 *      Author: josh
 */

#ifndef LSTM_H_
#define LSTM_H_


__global__ void lstm(float* in, float *out, float *W, float *B, float *U, long long N, long long M);
__host__ void execute_lstm(float *in, float *out, long long N, long long M, const cudaDeviceProp &cdp);
__host__ void execute_lstm_serial(float* in, float *out, long long N, long long M);


#endif /* LSTM_H_ */
