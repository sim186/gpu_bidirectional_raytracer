/**
 * @file rng_kernels.cu
 * @brief CUDA kernels for random number generation (Mersenne Twister).
 */

#include <stdio.h>
#include "mersenne_twister.h"

// Define constants from old macros if not defined
#ifndef MT_RNG_COUNT
#define MT_RNG_COUNT 4096
#endif

__device__ static mt_struct_stripped ds_MT[MT_RNG_COUNT];
static mt_struct_stripped h_MT[MT_RNG_COUNT];

/**
 * @brief Loads Mersenne Twister configurations from file.
 */
extern "C" void load_mt_gpu(const char* fname)
{
	FILE* fd = fopen(fname, "rb");
	if (!fd)
	{
		fprintf(stderr, "load_mt_gpu(): failed to open %s\n", fname);
		fprintf(stderr, "FAILED\n");
		exit(0);
	}
	if (!fread(h_MT, sizeof(h_MT), 1, fd))
	{
		fprintf(stderr, "load_mt_gpu(): failed to load %s\n", fname);
		fprintf(stderr, "FAILED\n");
		exit(0);
	}
	fclose(fd);
}

/**
 * @brief Initializes/seeds twister for current GPU context.
 */
extern "C" void seed_mt_gpu(unsigned int seed)
{
	int i;
	mt_struct_stripped* MT = (mt_struct_stripped*)malloc(MT_RNG_COUNT * sizeof(mt_struct_stripped));

	for (i = 0; i < MT_RNG_COUNT; i++)
	{
		MT[i] = h_MT[i];
		MT[i].seed = seed;
	}
	// Copy to constant symbol
	cudaMemcpyToSymbol(ds_MT, MT, sizeof(h_MT));

	free(MT);
}

/**
 * @brief Generates random numbers using Mersenne Twister.
 */
__global__ void random_gpu(
    float* d_random,
    int n_per_rng)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	int iState, iState1, iStateM, iOut;
	unsigned int mti, mti1, mtiM, x;
	unsigned int mt[MT_NN], matrix_a, mask_b, mask_c;

	// Load bit-vector Mersenne Twister parameters
	matrix_a = ds_MT[tid].matrix_a;
	mask_b = ds_MT[tid].mask_b;
	mask_c = ds_MT[tid].mask_c;

	// Initialize current state
	mt[0] = ds_MT[tid].seed;
	for (iState = 1; iState < MT_NN; iState++)
		mt[iState] = (1812433253U * (mt[iState - 1] ^ (mt[iState - 1] >> 30)) + iState) & MT_WMASK;

	iState = 0;
	mti1 = mt[0];
	for (iOut = 0; iOut < n_per_rng; iOut++)
	{
		iState1 = iState + 1;
		iStateM = iState + MT_MM;
		if (iState1 >= MT_NN)
			iState1 -= MT_NN;
		if (iStateM >= MT_NN)
			iStateM -= MT_NN;
		mti = mti1;
		mti1 = mt[iState1];
		mtiM = mt[iStateM];

		// MT recurrence
		x = (mti & MT_UMASK) | (mti1 & MT_LMASK);
		x = mtiM ^ (x >> 1) ^ ((x & 1) ? matrix_a : 0);

		mt[iState] = x;
		iState = iState1;

		// Tempering transformation
		x ^= (x >> MT_SHIFT0);
		x ^= (x << MT_SHIFTB) & mask_b;
		x ^= (x << MT_SHIFTC) & mask_c;
		x ^= (x >> MT_SHIFT1);

		// Convert to (0, 1] float and write to global memory
		d_random[tid + iOut * MT_RNG_COUNT] = ((float)x + 1.0f) / 4294967296.0f;
	}
}

/**
 * @brief Helper for Box-Muller transform.
 */
#define PI 3.14159265358979f
__device__ inline void box_muller(float& u1, float& u2)
{
	float r = sqrtf(-2.0f * logf(u1));
	float phi = 2 * PI * u2;
	u1 = r * __cosf(phi);
	u2 = r * __sinf(phi);
}

/**
 * @brief Transforms uniform random numbers to normal distribution using Box-Muller.
 */
__global__ void box_muller_gpu(float* d_random, int n_per_rng)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	for (int iOut = 0; iOut < n_per_rng; iOut += 2)
		box_muller(
		    d_random[tid + (iOut + 0) * MT_RNG_COUNT],
		    d_random[tid + (iOut + 1) * MT_RNG_COUNT]);
}
