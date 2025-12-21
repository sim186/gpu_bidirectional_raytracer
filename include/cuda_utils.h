#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

extern "C" void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

extern "C" int divide_round_up(int a, int b);

extern "C" int align_up(int a, int b);

#endif // CUDA_UTILS_H
