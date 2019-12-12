#pragma once

void printAllCUDASpecs(cudaDeviceProp& deviceProp);
void randomize(float* A, long long N, float low, float high);
long long readData(float **data, char *filename);
void lock();
void unlock();