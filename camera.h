#ifndef _CAMERA_H
#define	_CAMERA_H

//#include "cuPrintf.cuh"
//#include "cuPrintf.cu"
#include "vec.h"


typedef struct {
	/* User defined values */
	Vec orig, target;
	/* Calculated values */
	Vec dir, x, y;
} Camera;

#endif	/* _CAMERA_H */

