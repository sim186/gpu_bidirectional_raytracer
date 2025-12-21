#include "cuda_utils.h"

extern "C" void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

extern "C" int divide_round_up(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

extern "C" int align_up(int a, int b)
{
	return ((a % b) != 0) ? (a - a % b + b) : a;
}
