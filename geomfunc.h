
#ifndef _GEOMFUNC_H
#define	_GEOMFUNC_H

#include "geom.h"
#include "simplernd.h"
extern Camera camera, phantom;
extern uchar4 *pixels;
extern unsigned int *counter;
extern Vec *colors;
extern int width;
extern int height;
extern float invWidth;
extern float invHeight;
static float maxdiff=0;
int ccoo;
float ndep;

const char *byte_to_binary
(
    int x
)
{
    static char b[9];
    b[0] = '\0';

    int z;
    for (z = 256; z > 0; z >>= 1)
    {
        strcat(b, ((x & z) == z) ? "1" : "0");
    }

    return b;
}

static float SphereIntersect(
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
	const Sphere *s,
	const Ray *r) { /* returns distance, 0 if nohit */
	Vec op; /* Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 */
	vsub(op, s->p, r->o);

	float b = vdot(op, r->d);
	float det = b * b - vdot(op, op) + s->rad * s->rad;
	if (det < 0.f)
		return 0.f;
	else
		det = sqrt(det);

	float t = b - det;
	if (t >  EPSILON)
		return t;
	else {
		t = b + det;

		if (t >  EPSILON)
			return t;
		else
			return 0.f;
	}
}

static float CheckIntersect(
	const Sphere *s,
	const Ray *r) { /* returns distance, 0 if nohit */
	Vec op, ra; 
	vsub(op, s->p, r->o);
	vsub(ra,r->d, r->o);

	float len1 = sqrt(vdot(ra, ra));
	float len2 = sqrt(vdot(op, op));
	const float costheta = (vdot(ra,op)/(len1*len2));
	const float sentheta = sqrt(1-costheta*costheta);
	float d1 = len2 * sentheta;
	
	if (d1 < s->rad) {
		float d2 = len2 * costheta;
		return (d2-sqrt((s->rad)*(s->rad)-d1*d1));
	} else
		return 0.f;
	
}

static void UniformSampleSphere(const float u1, const float u2, Vec *v) {
	const float zz = 1.f - 2.f * u1;
	const float r = sqrt(max(0.f, 1.f - zz * zz));
	const float phi = 2.f * FLOAT_PI * u2;
	const float xx = r * cos(phi);
	const float yy = r * sin(phi);

	vinit(*v, xx, yy, zz);
}

static int Intersect2(
	const Sphere *spheres,
	const unsigned int sphereCount,
	const Ray *r,
	float *t,
	unsigned int *id) {
	float inf = (*t) = 1e20f;

	unsigned int i = sphereCount;
	for (; i--;) {
		const float d = CheckIntersect(&spheres[i], r);
		if ((d != 0.f) && (d < *t)) {
			*t = d;
			*id = i;
		}
	}

	return (*t < inf);
}

static int Intersect(
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
	const Sphere *spheres,
	const unsigned int sphereCount,
	const Ray *r,
	float *t,
	unsigned int *id) {
	float inf = (*t) = 1e20f;

	unsigned int i = sphereCount;
	for (; i--;) {
		const float d = SphereIntersect(&spheres[i], r);
		if ((d != 0.f) && (d < *t)) {
			*t = d;
			*id = i;
		}
	}

	return (*t < inf);
}

static int IntersectP(
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
	const Sphere *spheres,
	const unsigned int sphereCount,
	const Ray *r,
	const float maxt) {
	unsigned int i = sphereCount;
	for (; i--;) {
		const float d = SphereIntersect(&spheres[i], r);
		if ((d != 0.f) && (d < maxt))
			return 1;
	}

	return 0;
}

static void SampleLights(
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
	const Sphere *spheres,
	const unsigned int sphereCount,
	unsigned int *seed0, unsigned int *seed1,
	const Vec *hitPoint,
	const Vec *normal,
	Vec *result) {
	vclr(*result);

	/* For each light */
	unsigned int i;
	for (i = 0; i < sphereCount; i++) {
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
		const Sphere *light = &spheres[i];
		if (!viszero(light->e)) {
			/* It is a light source */
			Ray shadowRay;
			shadowRay.o = *hitPoint;

			/* Choose a point over the light source */
			Vec unitSpherePoint;
			UniformSampleSphere(GetRandom(seed0, seed1), GetRandom(seed0, seed1), &unitSpherePoint);
			Vec spherePoint;
			vsmul(spherePoint, light->rad, unitSpherePoint);
			vadd(spherePoint, spherePoint, light->p);

			/* Build the shadow ray direction */
			vsub(shadowRay.d, spherePoint, *hitPoint);
			const float len = sqrt(vdot(shadowRay.d, shadowRay.d));
			vsmul(shadowRay.d, 1.f / len, shadowRay.d);

			float wo = vdot(shadowRay.d, unitSpherePoint);
			if (wo > 0.f) {
				/* It is on the other half of the sphere */
				continue;
			} else
				wo = -wo;

			/* Check if the light is visible */
			const float wi = vdot(shadowRay.d, *normal);
			if ((wi > 0.f) && (!IntersectP(spheres, sphereCount, &shadowRay, len - EPSILON))) {
				Vec c; vassign(c, light->e);
				vsmul(c,10,c);
				const float s = (4.f * FLOAT_PI * light->rad * light->rad) * wi * wo / (len *len);
				//const float s = wi * wo / (len *len);
				vsmul(c, s, c);
				vadd(*result, *result, c);
			}
		}
	}
}


static void SamplePixels(
	const Vec *hitPoint,
	Vec *rad,  
	Vec *throughput,int test,
	Vec *normal, Ray prevRay) {

			/*float xp,yp;
			int pixx, pixy;
			Vec temp, emme, nega;
			unsigned int pixId;
			Ray shadowRay;
			shadowRay.o = *hitPoint;

			//vmul(*result,*throughput,obj->c);
			//vsmul(*result, M_PI, *result);
			//vadd(*result, *result, obj->e);
			Vec Ld;
			if(test==0){
				vsmul(Ld,M_PI,*throughput)
				vmul(*rad,*rad,Ld);
			}
			else{
			vsub(shadowRay.d, camera.orig, *hitPoint);
			const float len = sqrt(vdot(shadowRay.d, shadowRay.d));
			vsmul(shadowRay.d, 1.f / len, shadowRay.d);

			vxcross(temp,shadowRay.d,prevRay.d);
			float dotto=vdot(temp,*normal);
			//fprintf(stderr,"%f \n",dotto);
			if(fabs(dotto)<.01) {	
				const float len1 = sqrt(vdot(shadowRay.d, shadowRay.d));
				const float len2 = sqrt(vdot(*normal, *normal));
				const float len3 = sqrt(vdot(prevRay.d, prevRay.d));
				const float costheta1 = (vdot(shadowRay.d,*normal)/(len1*len2));
				const float costheta2 = (vdot(prevRay.d,*normal)/(len3*len2));
				//fprintf(stderr,"%f %f\n",costheta1,costheta2);
				if(fabs(costheta1-costheta2)<.01) {
					vsmul(Ld,M_PI,*throughput)
					vmul(*rad,*rad,Ld);
				} else {
					return;
				}
			} else return;
							
			}
			
			vassign(Ld,*rad);
			
			
			vsmul(nega,-1,camera.x);
			emme.x=vdot(nega,camera.orig);
			vsmul(nega,-1,camera.y);
			emme.y=vdot(nega,camera.orig);
			vsmul(nega,-1,camera.dir);
			emme.z=vdot(nega,camera.orig);
			temp.x=vdot(camera.x,*hitPoint)+emme.x;
			temp.y=vdot(camera.y,*hitPoint)+emme.y;
			temp.z=vdot(camera.dir,*hitPoint)+emme.z;
			if(temp.z>0){
			  xp=temp.x*10/temp.z;
			  yp=temp.y*10/temp.z;

			  const float Wf=invWidth*1.048*width/2.;
			  const float Hf=(invHeight*0.785*height/2.);
						
			  if ((xp<Wf)&&(xp>-Wf)&&(yp<(Hf-0))&&(yp>(-Hf+0))) {
				pixx=((xp+Wf)/(invWidth*1.048))+.5;
				pixy=((yp+Hf)/(invHeight*0.785)+.5);
				pixId=pixy*width+pixx;
				 if (counter[pixId]>0) 
					{
					Vec tmpc;
					vassign(tmpc,colors[pixId]);
					colors[pixId].x=(colors[pixId].x*counter[pixId]+Ld.x)/(counter[pixId]+1);
					if(fabs(colors[pixId].x-tmpc.x)>maxdiff)
						maxdiff=fabs(colors[pixId].x-tmpc.x);
					colors[pixId].y=(colors[pixId].y*counter[pixId]+Ld.y)/(counter[pixId]+1);
					if(fabs(colors[pixId].y-tmpc.y)>maxdiff)
						maxdiff=fabs(colors[pixId].y-tmpc.y);
					colors[pixId].z=(colors[pixId].z*counter[pixId]+Ld.z)/(counter[pixId]+1);
					if(fabs(colors[pixId].z-tmpc.z)>maxdiff)
						maxdiff=fabs(colors[pixId].z-tmpc.z);
				} else
					colors[pixId]=Ld;
				
					pixels[pixId]=toInt(colors[pixId].x)|(toInt(colors[pixId].y)<<8)|(toInt(colors[pixId].z)<<16);
				
				counter[pixId]+=1;
				
			   }
			}
		*/	
}

static void RadianceLightTracing(
	const Sphere *spheres,
	const unsigned int sphereCount,
	const Ray *startRay,
	unsigned int *seed0, unsigned int *seed1,
	const Sphere *inilight) {
	Ray currentRay; rassign(currentRay, *startRay);
	Vec rad; 
	Vec throughput; //vassign(throughput, rad);
	Sphere light; light=*inilight;
	vassign(rad, light.e);
	vassign(throughput, light.e);
	vnorm(throughput);
	vsmul(rad,1./2,rad);

	unsigned int depth = 1;
	for (;; ++depth) {
	//fprintf(stderr,"viszero r %d\n",viszero(light.e));
		// Removed Russian Roulette in order to improve execution on SIMT
		if (depth>50||viszero(rad)) {
			//if(ccoo<100)
			//fprintf(stderr,"%f <- %f - %d : %d ->",ndep*ccoo,ccoo,depth);
			ndep=(ndep*ccoo+depth)/(ccoo+1);
			//if(ccoo<100)
			//fprintf(stderr,"%f \n",ndep);
			ccoo++;
			return;
		}

		float t; /* distance to intersection */
		unsigned int id = 0; /* id of intersected object */
		if (!Intersect(spheres, sphereCount, &currentRay, &t, &id)) {
			return;
		}

		const Sphere *obj = &spheres[id]; /* the hit object */
		if(!viszero(obj->e)){
			return;
		}

		Vec hitPoint;
		vsmul(hitPoint, t, currentRay.d);
		vadd(hitPoint, currentRay.o, hitPoint);

		Vec normal;
		vsub(normal, hitPoint, obj->p);
		vnorm(normal);

		const float dp = vdot(normal, currentRay.d);

		Vec nl;
		// SIMT optimization
		const float invSignDP = -1.f * sign(dp);
		vsmul(nl, invSignDP, normal);

		if (obj->refl == DIFF) { /* Ideal DIFFUSE reflection */
			vmul(throughput, throughput, obj->c);
			//fprintf(stderr," col %f %f %f\n",throughput.x, throughput.y, throughput.z);

			/* Direct lighting component */

			unsigned int id2;
			Ray eyeray;
			eyeray.o=camera.orig;
			vsub(eyeray.d, hitPoint, camera.orig);
			const float len = sqrt(vdot(eyeray.d, eyeray.d));
			vsmul(eyeray.d, 1.f / len, eyeray.d);
			Intersect(spheres, sphereCount, &eyeray, &t, &id2);

			if(id2==id){

			//Vec Ld;
				
			SamplePixels(&hitPoint, &rad, &throughput,0,&normal,currentRay);
			}

			/* Diffuse component */

			float r1 = 2.f * FLOAT_PI * GetRandom(seed0, seed1);
			float r2 = GetRandom(seed0, seed1);
			float r2s = sqrt(r2);

			Vec w; vassign(w, nl);

			Vec u, a;
			if (fabs(w.x) > .1f) {
				vinit(a, 0.f, 1.f, 0.f);
			} else {
				vinit(a, 1.f, 0.f, 0.f);
			}
			vxcross(u, a, w);
			vnorm(u);

			Vec v;
			vxcross(v, w, u);

			Vec newDir;
			vsmul(u, cos(r1) * r2s, u);
			vsmul(v, sin(r1) * r2s, v);
			vadd(newDir, u, v);
			vsmul(w, sqrt(1 - r2), w);
			vadd(newDir, newDir, w);

			currentRay.o = hitPoint;
			currentRay.d = newDir;

			light=*obj;

			continue;
		} else if (obj->refl == SPEC) { /* Ideal SPECULAR reflection */

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

			vmul(throughput, throughput, obj->c);

			rinit(currentRay, hitPoint, newDir);
			continue;
		} else {

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

			Ray reflRay; rinit(reflRay, hitPoint, newDir); /* Ideal dielectric REFRACTION */
			int into = (vdot(normal, nl) > 0); /* Ray from outside going in? */

			float nc = 1.f;
			float nt = 1.5f;
			float nnt = into ? nc / nt : nt / nc;
			float ddn = vdot(currentRay.d, nl);
			float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.f)  { /* Total internal reflection */
				vmul(throughput, throughput, obj->c);

				rassign(currentRay, reflRay);
				continue;
			}

			float kk = (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t));
			Vec nkk;
			vsmul(nkk, kk, normal);
			Vec transDir;
			vsmul(transDir, nnt, currentRay.d);
			vsub(transDir, transDir, nkk);
			vnorm(transDir);

			float a = nt - nc;
			float b = nt + nc;
			float R0 = a * a / (b * b);
			float c = 1 - (into ? -ddn : vdot(transDir, normal));

			float Re = R0 + (1 - R0) * c * c * c * c*c;
			float Tr = 1.f - Re;
			float P = .25f + .5f * Re;
			float RP = Re / P;
			float TP = Tr / (1.f - P);

			if (GetRandom(seed0, seed1) < P) { /* R.R. */
				vsmul(throughput, RP, throughput);
				vmul(throughput, throughput, obj->c);

				rassign(currentRay, reflRay);
				continue;
			} else {
				vsmul(throughput, TP, throughput);
				vmul(throughput, throughput, obj->c);

				rinit(currentRay, hitPoint, transDir);
				continue;
			}
		}
	}
}

static void RadiancePathTracing(
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
	const Sphere *spheres,
	const unsigned int sphereCount,
	const Ray *startRay,
	unsigned int *seed0, unsigned int *seed1,
	Vec *result) {
	Ray currentRay; rassign(currentRay, *startRay);
	Vec rad; vinit(rad, 0.f, 0.f, 0.f);
	Vec throughput; vinit(throughput, 1.f, 1.f, 1.f);

	unsigned int depth = 0;
	int specularBounce = 1;
	for (;; ++depth) {
		// Removed Russian Roulette in order to improve execution on SIMT
		if (depth > 6) {
			*result = rad;
			return;
		}

		float t; /* distance to intersection */
		unsigned int id = 0; /* id of intersected object */
		if (!Intersect(spheres, sphereCount, &currentRay, &t, &id)) {
			*result = rad; /* if miss, return */
			return;
		}

#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
		const Sphere *obj = &spheres[id]; /* the hit object */

		Vec hitPoint;
		vsmul(hitPoint, t, currentRay.d);
		vadd(hitPoint, currentRay.o, hitPoint);

		Vec normal;
		vsub(normal, hitPoint, obj->p);
		vnorm(normal);

		const float dp = vdot(normal, currentRay.d);

		Vec nl;
		// SIMT optimization
		const float invSignDP = -1.f * sign(dp);
		vsmul(nl, invSignDP, normal);

		/* Add emitted light */
		Vec eCol; vassign(eCol, obj->e);
		if (!viszero(eCol)) {
			if (specularBounce) {
				vsmul(eCol, 10*fabs(dp), eCol);
				vmul(eCol, throughput, eCol);
				vadd(rad, rad, eCol);
			}

			*result = rad;
			return;
		}

		if (obj->refl == DIFF) { /* Ideal DIFFUSE reflection */
			specularBounce = 0;
			vmul(throughput, throughput, obj->c);

			/* Direct lighting component */

			Vec Ld;
			SampleLights(spheres, sphereCount, seed0, seed1, &hitPoint, &nl, &Ld);
			vmul(Ld, throughput, Ld);
			vadd(rad, rad, Ld);

			/* Diffuse component */

			float r1 = 2.f * FLOAT_PI * GetRandom(seed0, seed1);
			float r2 = GetRandom(seed0, seed1);
			float r2s = sqrt(r2);

			Vec w; vassign(w, nl);

			Vec u, a;
			if (fabs(w.x) > .1f) {
				vinit(a, 0.f, 1.f, 0.f);
			} else {
				vinit(a, 1.f, 0.f, 0.f);
			}
			vxcross(u, a, w);
			vnorm(u);

			Vec v;
			vxcross(v, w, u);

			Vec newDir;
			vsmul(u, cos(r1) * r2s, u);
			vsmul(v, sin(r1) * r2s, v);
			vadd(newDir, u, v);
			vsmul(w, sqrt(1 - r2), w);
			vadd(newDir, newDir, w);

			currentRay.o = hitPoint;
			currentRay.d = newDir;
			continue;
		} else if (obj->refl == SPEC) { /* Ideal SPECULAR reflection */
			specularBounce = 1;

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

			vmul(throughput, throughput, obj->c);

			rinit(currentRay, hitPoint, newDir);
			continue;
		} else {
			specularBounce = 1;

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

			Ray reflRay; rinit(reflRay, hitPoint, newDir); /* Ideal dielectric REFRACTION */
			int into = (vdot(normal, nl) > 0); /* Ray from outside going in? */

			float nc = 1.f;
			float nt = 1.5f;
			float nnt = into ? nc / nt : nt / nc;
			float ddn = vdot(currentRay.d, nl);
			float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.f)  { /* Total internal reflection */
				vmul(throughput, throughput, obj->c);

				rassign(currentRay, reflRay);
				continue;
			}

			float kk = (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t));
			Vec nkk;
			vsmul(nkk, kk, normal);
			Vec transDir;
			vsmul(transDir, nnt, currentRay.d);
			vsub(transDir, transDir, nkk);
			vnorm(transDir);

			float a = nt - nc;
			float b = nt + nc;
			float R0 = a * a / (b * b);
			float c = 1 - (into ? -ddn : vdot(transDir, normal));

			float Re = R0 + (1 - R0) * c * c * c * c*c;
			float Tr = 1.f - Re;
			float P = .25f + .5f * Re;
			float RP = Re / P;
			float TP = Tr / (1.f - P);

			if (GetRandom(seed0, seed1) < P) { /* R.R. */
				vsmul(throughput, RP, throughput);
				vmul(throughput, throughput, obj->c);

				rassign(currentRay, reflRay);
				continue;
			} else {
				vsmul(throughput, TP, throughput);
				vmul(throughput, throughput, obj->c);

				rinit(currentRay, hitPoint, transDir);
				continue;
			}
		}
	}
}

static void RadianceDirectLighting(
#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
	const Sphere *spheres,
	const unsigned int sphereCount,
	const Ray *startRay,
	unsigned int *seed0, unsigned int *seed1,
	Vec *result) {
	Ray currentRay; rassign(currentRay, *startRay);
	Vec rad; vinit(rad, 0.f, 0.f, 0.f);
	Vec throughput; vinit(throughput, 1.f, 1.f, 1.f);

	unsigned int depth = 0;
	int specularBounce = 1;
	for (;; ++depth) {
		// Removed Russian Roulette in order to improve execution on SIMT
		if (depth > 6) {
			*result = rad;
			return;
		}

		float t; /* distance to intersection */
		unsigned int id = 0; /* id of intersected object */
		if (!Intersect(spheres, sphereCount, &currentRay, &t, &id)) {
			*result = rad; /* if miss, return */
			return;
		}

#ifdef GPU_KERNEL
OCL_CONSTANT_BUFFER
#endif
		const Sphere *obj = &spheres[id]; /* the hit object */

		Vec hitPoint;
		vsmul(hitPoint, t, currentRay.d);
		vadd(hitPoint, currentRay.o, hitPoint);

		Vec normal;
		vsub(normal, hitPoint, obj->p);
		vnorm(normal);

		const float dp = vdot(normal, currentRay.d);

		Vec nl;
		// SIMT optimization
		const float invSignDP = -1.f * sign(dp);
		vsmul(nl, invSignDP, normal);

		/* Add emitted light */
		Vec eCol; vassign(eCol, obj->e);
		if (!viszero(eCol)) {
			if (specularBounce) {
				vsmul(eCol, fabs(dp), eCol);
				vmul(eCol, throughput, eCol);
				vadd(rad, rad, eCol);
			}

			*result = rad;
			return;
		}

		if (obj->refl == DIFF) { /* Ideal DIFFUSE reflection */
			specularBounce = 0;
			vmul(throughput, throughput, obj->c);

			/* Direct lighting component */

			Vec Ld;
			SampleLights(spheres, sphereCount, seed0, seed1, &hitPoint, &nl, &Ld);
			vmul(Ld, throughput, Ld);
			vadd(rad, rad, Ld);

			*result = rad;
			return;
		} else if (obj->refl == SPEC) { /* Ideal SPECULAR reflection */
			specularBounce = 1;

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

			vmul(throughput, throughput, obj->c);

			rinit(currentRay, hitPoint, newDir);
			continue;
		} else {
			specularBounce = 1;

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

			Ray reflRay; rinit(reflRay, hitPoint, newDir); /* Ideal dielectric REFRACTION */
			int into = (vdot(normal, nl) > 0); /* Ray from outside going in? */

			float nc = 1.f;
			float nt = 1.5f;
			float nnt = into ? nc / nt : nt / nc;
			float ddn = vdot(currentRay.d, nl);
			float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.f)  { /* Total internal reflection */
				vmul(throughput, throughput, obj->c);

				rassign(currentRay, reflRay);
				continue;
			}

			float kk = (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t));
			Vec nkk;
			vsmul(nkk, kk, normal);
			Vec transDir;
			vsmul(transDir, nnt, currentRay.d);
			vsub(transDir, transDir, nkk);
			vnorm(transDir);

			float a = nt - nc;
			float b = nt + nc;
			float R0 = a * a / (b * b);
			float c = 1 - (into ? -ddn : vdot(transDir, normal));

			float Re = R0 + (1 - R0) * c * c * c * c*c;
			float Tr = 1.f - Re;
			float P = .25f + .5f * Re;
			float RP = Re / P;
			float TP = Tr / (1.f - P);

			if (GetRandom(seed0, seed1) < P) { /* R.R. */
				vsmul(throughput, RP, throughput);
				vmul(throughput, throughput, obj->c);

				rassign(currentRay, reflRay);
				continue;
			} else {
				vsmul(throughput, TP, throughput);
				vmul(throughput, throughput, obj->c);

				rinit(currentRay, hitPoint, transDir);
				continue;
			}
		}
	}
}

#endif	/* _GEOMFUNC_H */

