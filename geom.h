#ifndef _GEOM_H
#define	_GEOM_H

#include "vec.h"

#define EPSILON 0.01f
#define FLOAT_PI 3.14159265358979323846f

typedef struct {
	Vec o, d;
} Ray;

#define rinit(r, a, b) { vassign((r).o, a); vassign((r).d, b); }
#define rassign(a, b) { vassign((a).o, (b).o); vassign((a).d, (b).d); }
#define LIGHT_POINTS 4096
#define DEPTH 1
#define MAX_VLP 1
#define MAX_ITER 3


enum Refl {
	DIFF, SPEC, REFR, LITE
}; /* material types, used in radiance() */

typedef struct {
	float rad; /* radius */
	Vec p, e, c; /* position, emission, color */
	enum Refl refl; /* reflection type (DIFFuse, SPECular, REFRactive) */
} Sphere;

typedef struct {
	Vec hp;
	Vec rad;
	Vec nl;
} LightPath;

#endif	/* _GEOM_H */

