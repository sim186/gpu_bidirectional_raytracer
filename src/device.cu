#include "vec.h"
#include "geom.h"
#include "camera.h"
#include <math.h>
#include <stdio.h>
#include "cons.h"

__device__ static float tol=TOL;

__device__ static bool vecmul(Vec *inout, Vec mul) {

bool should_stop = false;
float temp;

if (inout->x!=0.f&&mul.x!=0.f) {
	temp=inout->x*mul.x;
	if (temp<=tol||inout->x==temp)
		should_stop=true;
	else
	 	inout->x=temp;
	}
else
	inout->x=0.f;

if (inout->y!=0.f&&mul.y!=0.f) {
	temp=inout->y*mul.y;
	if (temp<=tol||inout->y==temp)
		should_stop=true;
	else
	 	inout->y=temp;
	}
else
	inout->y=0.f;

if (inout->z!=0.f&&mul.z!=0.f) {
	temp=inout->z*mul.z;
	if (temp<=tol||inout->z==temp)
		should_stop=true;
	else
	 	inout->z=temp;
	}
else
	inout->z=0.f;
return should_stop;
}


__device__ static bool vecsmul(Vec *inout, float mul) {

bool should_stop = false;
float temp;

if (inout->x!=0.f&&mul!=0.f) {
	temp=inout->x*mul;
	if (temp<=tol||inout->x==temp)
		should_stop=true;
	else
	 	inout->x=temp;
	}
else
	inout->x=0.f;

if (inout->y!=0.f&&mul!=0.f) {
	temp=inout->y*mul;
	if (temp<=tol||inout->y==temp)
		should_stop=true;
	else
	 	inout->y=temp;
	}
else
	inout->y=0.f;

if (inout->z!=0.f&&mul!=0.f) {
	temp=inout->z*mul;
	if (temp<=tol||inout->z==temp)
		should_stop=true;
	else
	 	inout->z=temp;
	}
else
	inout->z=0.f;
return should_stop;
}


__device__ static float SphereIntersect_dev(
	Sphere *s,
	Ray *r) { /* returns distance, 0 if nohit */
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

__device__ static int Intersect_dev(
	Sphere *spheres,
	unsigned int sphereCount,
	Ray *r,
	float *t,
	unsigned int *id) {
	float inf = (*t) = 1e20f;

	unsigned int i = sphereCount;
	for (; i--;) {
		const float d = SphereIntersect_dev(&spheres[i], r);
		if ((d != 0.f) && (d < *t)) {
			*t = d;
			*id = i;
		}
	}

	return (*t < inf);
}

__device__ static int IntersectP_dev(
	Sphere *spheres,
	unsigned int sphereCount,
	Ray *r,
	float maxt) {
	unsigned int i = sphereCount;
	for (; i--;) {
		const float d = SphereIntersect_dev(&spheres[i], r);
		if ((d != 0.f) && (d < maxt))
			return 1;
	}

	return 0;
}

__device__ static int IntersectP_vaacum_dev(
	Sphere *spheres,
	unsigned int sphereCount,
	Ray *r,
	float maxt) {
	unsigned int i = sphereCount;
	for (; i--;) {
		const float d = SphereIntersect_dev(&spheres[i], r);
		if ((d != 0.f) && (d < maxt) && viszero(spheres[i].e))
			return 1;
	}

	return 0;
}


__device__ static void UniformSampleSphere_dev(const float u1, const float u2, Vec *v) {
	const float zz = 1.f - 2.f * u1;
	const float r = sqrt(max(0.f, 1.f - zz * zz));
	const float phi = 2.f * FLOAT_PI * u2;
	const float xx = r * cos(phi);
	const float yy = r * sin(phi);

	vinit(*v, xx, yy, zz);
}

__global__ void get_ray(const Sphere light, Ray *dev_data, float *d_Rand, int cS, int pC, unsigned int sid)
{
	/* Choose a point over the light source */
	int i,j;
	int ind=blockDim.x*blockIdx.x+threadIdx.x;

	i=(cS*5+ind*4+sid)%(pC-4);
	j=i+2;
	//j=(lS+(ind*2))%(pC-4);

	Vec unitSpherePoint;
	UniformSampleSphere_dev(d_Rand[j], d_Rand[i], &unitSpherePoint);
	Vec spherePoint;
	vsmul(spherePoint, light.rad, unitSpherePoint);
	vadd(spherePoint, spherePoint, light.p);

	/* Build the newray direction */
	Vec normal;
	vsub(normal, spherePoint, light.p);
	vnorm(normal);

	//vsmul(normal, 1.f, normal);

	float r1 = 2.f * FLOAT_PI * d_Rand[i+1];
	float r2 = d_Rand[j+1];
	float r2s = sqrt(r2);

	Vec u, a;

	if (fabs(normal.x) > .1f) {
		vinit(a, 0.f, 1.f, 0.f);
	} else {
		vinit(a, 1.f, 0.f, 0.f);
	}
	vxcross(u, a, normal);
	vnorm(u);

	Vec v;
	vxcross(v, normal, u);

	Vec newDir;
	vsmul(u, cos(r1) * r2s, u);
	vsmul(v, sin(r1) * r2s, v);
	vadd(newDir, u, v);
	vsmul(normal, sqrt(1 - r2), normal);
	vadd(newDir, newDir, normal);

	/*dev_data[ind].o.x = dev_seeds[i];
	dev_data[ind].o.y = i;
	dev_data[ind].o.z = lS;*/
	dev_data[ind].o = spherePoint;
	dev_data[ind].d = newDir;
}


__global__ void RadianceLightTracing_dev(
           Sphere *spheres, 
           unsigned int sphereCount, 
           Ray *startRay, 
           float *d_Rand, 
           int idlight, 
           int pC, 
           Camera camera, 
           Vec *colors, 
           unsigned int *counter, 
           LightPath *dev_lp, 
           float invWidth, 
           float invHeight, 
           float width, 
           float height, 
           unsigned int sid) {

	int ind=blockDim.x*blockIdx.x+threadIdx.x;
	int row=blockDim.y*blockIdx.y+threadIdx.y;

	bool flag1=false, flag2=false;
	Sphere inilight=spheres[idlight];
	int depth_index,depth=0;
	
	Ray currentRay; rassign(currentRay, startRay[ind]);
	Vec throughput;
	vassign(throughput, inilight.e);
	Vec hitPoint;
	vassign(hitPoint, currentRay.o);

	Vec normal;
	vsub(normal, hitPoint, inilight.p);
	vnorm(normal);
	{	
			unsigned int id2;
			Ray eyeray;
			float t;
			eyeray.o=camera.orig;
			vsub(eyeray.d, hitPoint, camera.orig);
			const float len = sqrt(vdot(eyeray.d, eyeray.d));
			vsmul(eyeray.d, 1.f / len, eyeray.d);
			Intersect_dev(spheres, sphereCount, &eyeray, &t, &id2);

	}
	
	//serve?
	vsmul(throughput, 1./4, throughput);
	
	
	while (depth<DEPTH) { //Aggiungere define DEPTH?
	
		int i=(ind*154+depth*3+4+sid)%(pC-3);
		//lS*155+
		
		// Removed Russian Roulette in order to improve execution on SIMT
		float t; /* distance to intersection */
		unsigned int id = 0; /* id of intersected object */
		if (!Intersect_dev(spheres, sphereCount, &currentRay, &t, &id)) {
			Vec nor;
			vsub(nor, currentRay.o,inilight.p);
			vsmul(nor,-1./inilight.rad,nor);
			depth_index = depth*LIGHT_POINTS+ind;
		    dev_lp[depth_index].hp.x = currentRay.o.x;
		    dev_lp[depth_index].hp.y = currentRay.o.y;
		    dev_lp[depth_index].hp.z = currentRay.o.z;
            Vec inilight_va;
            vsmul(inilight_va,1./2,inilight.e);
		    dev_lp[depth_index].rad = inilight_va;
		    dev_lp[depth_index].nl= nor;
			
			return ;
		}

		const Sphere *obj = &spheres[id]; /* the hit object */
		if(!viszero(obj->e)){
			return ;
		}

		vsmul(hitPoint, t, currentRay.d);
		vadd(hitPoint, currentRay.o, hitPoint);

		vsub(normal, hitPoint, obj->p);
		vnorm(normal);

		const float dp = vdot(normal, currentRay.d);

		Vec nl;
		// SIMT optimization
		const float invSignDP = -1.f * sign(dp);
		vsmul(nl, invSignDP, normal);


		if (obj->refl == DIFF) { /* Ideal DIFFUSE reflection */
			// if(||vecsmul(&throughput, 0.1))
				// flag1=true;
             vecmul(&throughput, obj->c);
			/* Direct lighting component */

			// unsigned int id2;
			// Ray eyeray;
			// eyeray.o=camera.orig;
			// vsub(eyeray.d, hitPoint, camera.orig);
			// const float len = sqrt(vdot(eyeray.d, eyeray.d));
			// vsmul(eyeray.d, 1.f / len, eyeray.d);
			// Intersect_dev(spheres, sphereCount, &eyeray, &t, &id2);


//&&(len<=(t+0.001)
                if(depth < DEPTH){
					depth_index = depth*LIGHT_POINTS+ind;
					// depth_index = row*DEPTH + depth;
				    dev_lp[depth_index].hp.x = hitPoint.x;
				    dev_lp[depth_index].hp.y = hitPoint.y;
				    dev_lp[depth_index].hp.z = hitPoint.z;
				    dev_lp[depth_index].rad = throughput;
				    dev_lp[depth_index].nl= nl;

				    // dev_lp[depth_index].hp.x = depth_index;
				    // dev_lp[depth_index].hp.y = depth;
				    // dev_lp[depth_index].hp.z = DEPTH;
				    
				    
				    //dev_lp[depth_index].rad.y = throughput.y;
				    //dev_lp[depth_index].rad.z = throughput.z;
				}
				else { break;}
                                      
                   
                   //cuPrintf("x=%.3f y=%.3f z=%.3f \n",hitPoint.x,hitPoint.y,hitPoint.z);
                   
			//}

			/* Diffuse component */


			float r1 = 2.f * FLOAT_PI * d_Rand[i];
			float r2 = d_Rand[i+1];
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

		} else if (obj->refl == SPEC) { /* Ideal SPECULAR reflection */

			Vec newDir;
			vsmul(newDir,  2.f * vdot(normal, currentRay.d), normal);
			vsub(newDir, currentRay.d, newDir);

            vecmul(&throughput, obj->c);
			// if ()
				// flag1=true;

			rinit(currentRay, hitPoint, newDir);
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
				
				vecmul(&throughput, obj->c);
				// if ()
					// flag1=true;

				rassign(currentRay, reflRay);
			} else {

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

				if (d_Rand[i+2] < P) { /* R.R. */
					if (vecsmul(&throughput, RP)||vecmul(&throughput, obj->c))
						flag1=true;

					rassign(currentRay, reflRay);
				} else {
					if (vecsmul(&throughput, TP)||vecmul(&throughput, obj->c))
						flag1=true;

					rinit(currentRay, hitPoint, transDir);
				}
			}
		}
		depth++;
		
	}
}

__device__ static void SampleLights_dev(
	Sphere *spheres,
	unsigned int sphereCount,
	float *d_Rand,
	Vec *hitPoint,
	Vec *normal,
	Vec *result, int pC, int nn,LightPath *dev_lp,int vlp_index) {
	vclr(*result);
	
/* For each light */
	unsigned int i,j,k,dk,dj;
	dk=nn+1;
	dj=nn+2;
	for (i = 0; i < sphereCount; i++) {
     const Sphere *light = &spheres[i];
		if (!viszero(light->e)) {
			/* It is a light source */
			Ray shadowRay;
			shadowRay.o = *hitPoint;

			/* Choose a point over the light source */
			Vec unitSpherePoint;
			UniformSampleSphere_dev(d_Rand[dk], d_Rand[dj], &unitSpherePoint);
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
			if ((wi > 0.f) && (!IntersectP_dev(spheres, sphereCount, &shadowRay, len - EPSILON))) {
				Vec c; vassign(c, light->e);
				//vsmul(c,4,c);
				const float s = (4.f * FLOAT_PI * light->rad * light->rad) * wi * wo / (len *len);
				//const float s = wi * wo / (len *len);
				vsmul(c, s, c);
				vadd(*result, *result, c);
			}
		}
	}
	Ray shadowRayVirtual;
	Vec VLP_result;
	vclr(VLP_result);
    for(j = vlp_index; j < vlp_index+MAX_VLP; j++){
        for(k = 0; k < DEPTH; k++){
				shadowRayVirtual.o = *hitPoint;
				//Vec VirtualLightsPoint = dev_lp[k][j].hp;
                Vec VirtualLightsPoint = dev_lp[k*LIGHT_POINTS+j].hp;

				vsub(shadowRayVirtual.d, VirtualLightsPoint, *hitPoint);
				/* Build the shadow ray direction */
				const float len = sqrt(vdot(shadowRayVirtual.d, shadowRayVirtual.d));
				vsmul(shadowRayVirtual.d, 1.f / len, shadowRayVirtual.d);
				float wo = vdot(shadowRayVirtual.d, dev_lp[j+k*LIGHT_POINTS].nl);
				if (wo > 0.f) {
					/* It is on the other half of the sphere */
					continue;
				} else
					wo = -wo;
				/* Check if the light is visible */
				const float wi = vdot(shadowRayVirtual.d, *normal);
				if ((wi > 0.f) && (!IntersectP_vaacum_dev(spheres, sphereCount, &shadowRayVirtual, len - EPSILON))) {
					Vec c; vassign(c, dev_lp[k*LIGHT_POINTS+j].rad);

					//const float s = 0.1;
					const float s = wi*wo;
					vsmul(c, s, c);
					vadd(VLP_result, VLP_result, c);
				}
      }/*for j*/
    }/*for k*/
    vsmul(VLP_result,1./(DEPTH*MAX_VLP),VLP_result);
    vadd(*result, *result, VLP_result);
    vsmul(*result,1./2,*result);
	
}

__global__ void RadiancePathTracing_dev(
	Sphere *spheres,
	unsigned int sphereCount,
	float *d_Rand,
	int pC, Camera camera, Vec *colors, unsigned int *counter,
	uchar4 *pixels,float invWidth, float invHeight,
	float width, float height, 
	unsigned int sid, unsigned int param,LightPath *dev_lp, int vlp_index) {

	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;

	if (x<=width) { //se TID del thread del device è minore dell'asse x del piano immagine, in sostanza sono finiti i pixel da assegnare
		int i=y*width+x; //indice del pixel
		Vec r;
	
		//Ray ray;
		Vec rdir,temp,nega, kappa;
		int kk=(26+i*25+sid)%(pC-5);
		// const float kx=((float)(x)*(invWidth)-invWidth*width/2.);
		// const float ky=((float)(y)*(invHeight)-invHeight*height/2.);
		const float kx=((float)(x)*(invWidth)-invWidth*width/2.)+d_Rand[kk]*invWidth;
		const float ky=((float)(y)*(invHeight)-invHeight*height/2.)+d_Rand[kk+1]*invHeight;
		const float kz=10.0;
		vinit (kappa,kx,ky,kz);
			
		vinit(rdir,0.f,0.f,0.f);
		/*dati sul punto di osservazione*/
		vassign(temp,camera.x);
		vnorm(temp);
		vsmul(temp,kx,temp);
		vadd(rdir,rdir,temp);
		vassign(temp,camera.y);
		vnorm(temp);
		vsmul(temp,ky,temp);
		vadd(rdir,rdir,temp);
		vassign(temp,camera.dir);
		vnorm(temp);
		vsmul(temp,kz,temp);
		vadd(rdir,rdir,temp);
		vsmul(nega,-1,camera.x);
		vnorm(nega);
		temp.x=vdot(nega,camera.orig);
		vsmul(nega,-1,camera.y);
		vnorm(nega);
		temp.y=vdot(nega,camera.orig);
		vnorm(nega);
		vsmul(nega,-1,camera.dir);
		temp.z=vdot(nega,camera.orig);
		float w=vdot(temp,kappa)+1;
		vsmul(rdir,1./w,rdir);
			
		Vec rorig;
		vadd(rorig, rdir, camera.orig);
	
		vnorm(rdir);
		Ray ray = {rorig, rdir};
				
		unsigned int id2;
		float t;
		Intersect_dev(spheres, sphereCount, &ray, &t, &id2);
		int nn=1;
	
		if(counter[i]<30000){//A che serve? Sarebbe il numero di percorsi per tutta l'immagine?
	
		   Ray currentRay; rassign(currentRay, ray);
		   Vec rad; vinit(rad, 0.f, 0.f, 0.f);
	  	   Vec throughput; vinit(throughput, 1.f, 1.f, 1.f);
	       //vclr(*result);
	
		   unsigned int depth = 0;
		   int specularBounce = 1;
		   for (;; ++depth) {
				//int j=(i+eyeSample+depth*(pC/6))%(pC/2);
				//int j=(i*eyeSample*nn*depth)%(pC-4);
				int j=(nn*26+i*25+depth*5+sid)%(pC-5);
				// Removed Russian Roulette in order to improve execution on SIMT
				if (depth > 6) {
					r = rad;
					break;
				}
		
				float t; /* distance to intersection */
				unsigned int id = 0; /* id of intersected object */
				if (!Intersect_dev(spheres, sphereCount, &currentRay, &t, &id)) {
					r = rad; /* if miss, return */
					break;
				}
		
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
		
					 r = rad;
					 break;
				 }
		
				if (obj->refl == DIFF) { /* Ideal DIFFUSE reflection */
					specularBounce = 0;
					vmul(throughput, throughput, obj->c);
		
					/* Direct lighting component */
		
					Vec Ld;
					SampleLights_dev(spheres, sphereCount, d_Rand, &hitPoint, &nl, &Ld, pC,j+2,dev_lp,vlp_index);
					vmul(Ld, throughput, Ld);
					vadd(rad, rad, Ld);
		
					/* Diffuse component */
		
					float r1 = 2.f * FLOAT_PI * d_Rand[j];
					float r2 = d_Rand[j+1];
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
		
					if (d_Rand[j+2] < P) { /* R.R. */
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
		   } //for depth
		
		
		  if(counter[i] ==0){ colors[i]=r;}
		  else
		  {
			  const float k1 = counter[i];
			  const float k2 = 1.f / (k1 + 1.f);
			  colors[i].x = (colors[i].x * k1 + r.x) * k2;
			  colors[i].y = (colors[i].y * k1 + r.y) * k2;
			  colors[i].z = (colors[i].z * k1 + r.z) * k2;
		  }	
		   pixels[i].w=0;
		   pixels[i].x=toInt(colors[i].x);
		   pixels[i].y=toInt(colors[i].y);
		   pixels[i].z=toInt(colors[i].z);
		   counter[i]+=1;
		   nn++;
		}//counter
    }
}
