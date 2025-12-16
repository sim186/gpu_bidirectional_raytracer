#pragma once

// Constants for ray tracing
#define RAYNTHREAD 64
#define RAYNGRID 64

// Total number of threads
#define TOTTHREAD (RAYNTHREAD * RAYNGRID)

// Maximum number of iterations for rendering
#define MAXITER 6

// Tolerance for floating point comparisons
#define TOL 0.0001
