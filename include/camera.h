#pragma once
#include "vector_math.h"

// Camera structure
typedef struct
{
	Vec orig, target;
	Vec dir, x, y;
} Camera;
