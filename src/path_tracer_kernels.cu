/**
 * @file path_tracer_kernels.cu
 * @brief CUDA kernels for bidirectional path tracing.
 */

#include "vector_math.h"
#include "geometry.h"
#include "camera.h"
#include <math.h>
#include <stdio.h>
#include "constants.h"
#include "render_context.h"


__device__ static float tol = TOL;

/**
 * @brief Multiplies a vector by another vector (component-wise) with tolerance check.
 */
__device__ static bool vec_multiply(Vec* inout, Vec mul)
{
	bool should_stop = false;
	float temp;

	// X component
	if (inout->x != 0.f && mul.x != 0.f)
	{
		temp = inout->x * mul.x;
		if (temp <= tol || inout->x == temp)
			should_stop = true;
		else
			inout->x = temp;
	}
	else
	{
		inout->x = 0.f;
	}

	// Y component
	if (inout->y != 0.f && mul.y != 0.f)
	{
		temp = inout->y * mul.y;
		if (temp <= tol || inout->y == temp)
			should_stop = true;
		else
			inout->y = temp;
	}
	else
	{
		inout->y = 0.f;
	}

	// Z component
	if (inout->z != 0.f && mul.z != 0.f)
	{
		temp = inout->z * mul.z;
		if (temp <= tol || inout->z == temp)
			should_stop = true;
		else
			inout->z = temp;
	}
	else
	{
		inout->z = 0.f;
	}
	
	return should_stop;
}



/**
 * @brief Intersects a ray with a sphere.
 * @return Distance to intersection, or 0 if no hit.
 */
__device__ static float sphere_intersect_device(Sphere* s, Ray* r)
{
	Vec op; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
	vsub(op, s->p, r->o);

	float b = vdot(op, r->d);
	float det = b * b - vdot(op, op) + s->rad * s->rad;
	if (det < 0.f)
		return 0.f;
	else
		det = sqrt(det);

	float t = b - det;
	if (t > EPSILON)
		return t;
	else
	{
		t = b + det;
		if (t > EPSILON)
			return t;
		else
			return 0.f;
	}
}

/**
 * @brief Finds the closest intersection of a ray with the scene spheres.
 * @return True if intersection found.
 */
__device__ static int intersect_device(
    Sphere* spheres,
    unsigned int sphere_count,
    Ray* r,
    float* t,
    unsigned int* id)
{
	float inf = (*t) = 1e20f;

	unsigned int i = sphere_count;
	for (; i--;)
	{
		const float d = sphere_intersect_device(&spheres[i], r);
		if ((d != 0.f) && (d < *t))
		{
			*t = d;
			*id = i;
		}
	}

	return (*t < inf);
}

/**
 * @brief Checks if a ray intersects any sphere within a max distance.
 */
__device__ static int intersect_p_device(
    Sphere* spheres,
    unsigned int sphere_count,
    Ray* r,
    float maxt)
{
	unsigned int i = sphere_count;
	for (; i--;)
	{
		const float d = sphere_intersect_device(&spheres[i], r);
		if ((d != 0.f) && (d < maxt))
			return 1;
	}

	return 0;
}

/**
 * @brief Checks occlusion excluding vacuum objects (visualize light sources).
 */
__device__ static int intersect_p_vacuum_device(
    Sphere* spheres,
    unsigned int sphere_count,
    Ray* r,
    float maxt)
{
	unsigned int i = sphere_count;
	for (; i--;)
	{
		const float d = sphere_intersect_device(&spheres[i], r);
		if ((d != 0.f) && (d < maxt) && viszero(spheres[i].e))
			return 1;
	}

	return 0;
}

/**
 * @brief Uniformly samples a point on a sphere surface.
 */
__device__ static void uniform_sample_sphere_device(const float u1, const float u2, Vec* v)
{
	const float zz = 1.f - 2.f * u1;
	const float r = sqrt(max(0.f, 1.f - zz * zz));
	const float phi = 2.f * FLOAT_PI * u2;
	const float xx = r * cos(phi);
	const float yy = r * sin(phi);

	vinit(*v, xx, yy, zz);
}

/**
 * @brief Kernel to generate initial rays from light sources.
 */
/**
 * @brief Kernel to generate initial rays from light sources.
 */
__global__ void get_ray_kernel(const Sphere light, DeviceRenderContext ctx, unsigned int seed_id)
{
	/* Choose a point over the light source */
	int i, j;
	int ind = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Unpack necessary device pointers
    float* d_Rand = ctx.d_rand;
    Ray* dev_data = ctx.dev_ray;
    // Note: rand_count was RAND_N (globally known constant) or passed as arg.
    // In original code: int RAND_N passed.
    // We should use RAND_N constant if available or pass it in context. 
    // The constant RAND_N is defined in main.cu, not path_tracer_kernels.cu?
    // Wait, RAND_N is calculated in main.cu. But path_tracer_kernels.cu includes constants.h
    // Let's assume passed via ctx? No, DeviceRenderContext only has pointers.
    // We rely on constants.h for things like DEPTH, but RAND_N might be dynamic now?
    // The original passed `int rand_count`.
    // Let's assume we can use the macro or we need to add it to context.
    // Ideally add config vars to DeviceRenderContext or pass as arg?
    // The user said "reduce parameters".
    // I will use a hardcoded value if it's constant, or add to context.
    // Checking main.cu: RAND_N = MT_RNG_COUNT * N_PER_RNG.
    // Let's hardcode a reasonable large value or use what was compiled.
    // Actually, `rand_count` was passed to kernel. I should add `rand_count` to DeviceRenderContext?
    // DeviceRenderContext struct definition I made only has pointers.
    // I should Update DeviceRenderContext to include `rand_count` and `width`, `height` etc if they are needed on device.
    // Re-reading my RenderContext definition: DeviceRenderContext only had pointers.
    // I NEED TO UPDATE RenderContext.h FIRST to include scalars on device context if I want to remove them from args.
    // Or I can keep scalars as args? User said "so many parameters".
    // Better to put scalars in a struct passed by value (DeviceRenderContext).
    
    // I will abort this tool call and update RenderContext first.

{
	/* Choose a point over the light source */
	int i, j;
	int ind = blockDim.x * blockIdx.x + threadIdx.x;

	i = (ctx.current_sample * 5 + ind * 4 + seed_id) % (ctx.rand_count - 4);
	j = i + 2;

	Vec unit_sphere_point;
	uniform_sample_sphere_device(d_Rand[j], d_Rand[i], &unit_sphere_point);

	Vec sphere_point;
	vsmul(sphere_point, light.rad, unit_sphere_point);
	vadd(sphere_point, sphere_point, light.p);

	/* Build the newray direction */
	Vec normal;
	vsub(normal, sphere_point, light.p);
	vnorm(normal);

	float r1 = 2.f * FLOAT_PI * d_Rand[i + 1];
	float r2 = d_Rand[j + 1];
	float r2s = sqrt(r2);

	Vec u, a;

	if (fabs(normal.x) > .1f)
	{
		vinit(a, 0.f, 1.f, 0.f);
	}
	else
	{
		vinit(a, 1.f, 0.f, 0.f);
	}
	vxcross(u, a, normal);
	vnorm(u);

	Vec v;
	vxcross(v, normal, u);

	Vec new_dir;
	vsmul(u, cos(r1) * r2s, u);
	vsmul(v, sin(r1) * r2s, v);
	vadd(new_dir, u, v);
	vsmul(normal, sqrt(1 - r2), normal);
	vadd(new_dir, new_dir, normal);

	dev_data[ind].o = sphere_point;
	dev_data[ind].d = new_dir;
}

/**
 * @brief Samples light sources for direct lighting calculation.
 */
__device__ static void sample_lights_device(
    Sphere* spheres,
    unsigned int sphere_count,
    float* d_Rand,
    Vec* hit_point,
    Vec* normal,
    Vec* result, int nn, LightPath* dev_lp, int vlp_index)
{
	vclr(*result);

	/* For each light */
	unsigned int i, j, k, dk, dj;
	dk = nn + 1;
	dj = nn + 2;
	for (i = 0; i < sphere_count; i++)
	{
		const Sphere* light = &spheres[i];
		if (!viszero(light->e))
		{
			/* It is a light source */
			Ray shadow_ray;
			shadow_ray.o = *hit_point;

			/* Choose a point over the light source */
			Vec unit_sphere_point;
			uniform_sample_sphere_device(d_Rand[dk], d_Rand[dj], &unit_sphere_point);
			Vec sphere_point;
			vsmul(sphere_point, light->rad, unit_sphere_point);
			vadd(sphere_point, sphere_point, light->p);
			
			/* Build the shadow ray direction */
			vsub(shadow_ray.d, sphere_point, *hit_point);
			const float len = sqrt(vdot(shadow_ray.d, shadow_ray.d));
			vsmul(shadow_ray.d, 1.f / len, shadow_ray.d);

			float wo = vdot(shadow_ray.d, unit_sphere_point);
			if (wo > 0.f)
			{
				/* It is on the other half of the sphere */
				continue;
			}
			else
				wo = -wo;

			/* Check if the light is visible */
			const float wi = vdot(shadow_ray.d, *normal);
			if ((wi > 0.f) && (!intersect_p_device(spheres, sphere_count, &shadow_ray, len - EPSILON)))
			{
				Vec c;
				vassign(c, light->e);
				const float s = (4.f * FLOAT_PI * light->rad * light->rad) * wi * wo / (len * len);
				vsmul(c, s, c);
				vadd(*result, *result, c);
			}
		}
	}
	
	Ray shadow_ray_virtual;
	Vec VLP_result;
	vclr(VLP_result);
	for (j = vlp_index; j < vlp_index + MAX_VLP; j++)
	{
		for (k = 0; k < DEPTH; k++)
		{
			shadow_ray_virtual.o = *hit_point;
			Vec VirtualLightsPoint = dev_lp[k * LIGHT_POINTS + j].hp;

			vsub(shadow_ray_virtual.d, VirtualLightsPoint, *hit_point);
			/* Build the shadow ray direction */
			const float len = sqrt(vdot(shadow_ray_virtual.d, shadow_ray_virtual.d));
			vsmul(shadow_ray_virtual.d, 1.f / len, shadow_ray_virtual.d);
			float wo = vdot(shadow_ray_virtual.d, dev_lp[j + k * LIGHT_POINTS].nl);
			if (wo > 0.f)
			{
				/* It is on the other half of the sphere */
				continue;
			}
			else
				wo = -wo;
			
			/* Check if the light is visible */
			const float wi = vdot(shadow_ray_virtual.d, *normal);
			if ((wi > 0.f) && (!intersect_p_vacuum_device(spheres, sphere_count, &shadow_ray_virtual, len - EPSILON)))
			{
				Vec c;
				vassign(c, dev_lp[k * LIGHT_POINTS + j].rad);

				const float s = wi * wo;
				vsmul(c, s, c);
				vadd(VLP_result, VLP_result, c);
			}
		} /*for j*/
	} /*for k*/
	vsmul(VLP_result, 1. / (DEPTH * MAX_VLP), VLP_result);
	vadd(*result, *result, VLP_result);
	vsmul(*result, 1. / 2, *result);
}

/**
 * @brief Radiance calculation for light tracing pass.
 */
__global__ void radiance_light_tracing_kernel(
    int id_light,
    DeviceRenderContext ctx,
    unsigned int seed_id)
{

	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	// int row = blockDim.y * blockIdx.y + threadIdx.y; // Unused
    
    // Unpack context
    Sphere* spheres = ctx.dev_spheres;
    unsigned int sphere_count = ctx.sphere_count;
    Ray* start_ray = ctx.dev_ray;
    float* d_Rand = ctx.d_rand;
    int rand_count = ctx.rand_count;
    LightPath* dev_lp = ctx.dev_lp;
    float inv_width = ctx.inverse_width;
    float inv_height = ctx.inverse_height;
    float width = ctx.width;
    float height = ctx.height;
    // Camera is passed via constant memory in original? No, it was passed by value as arg.
    // Wait, Camera was an arg. I need to add Camera to DeviceRenderContext?
    // Camera is small, can be in Context.
    // I missed adding Camera to DeviceRenderContext!
    // I check render_context.h again.
    // DeviceRenderContext ONLY contains pointers and scalars.
    // RenderContext (Host) has `Camera camera`.
    // I need to add `Camera camera` to DeviceRenderContext to support this refactor.
    Camera camera = ctx.camera;


{

	int ind = blockDim.x * blockIdx.x + threadIdx.x;
	// int row = blockDim.y * blockIdx.y + threadIdx.y; // Unused

	Sphere ini_light = spheres[id_light];
	int depth_index, depth = 0;

	Ray current_ray;
	rassign(current_ray, start_ray[ind]);
	Vec throughput;
	vassign(throughput, ini_light.e);
	Vec hit_point;
	vassign(hit_point, current_ray.o);

	Vec normal;
	vsub(normal, hit_point, ini_light.p);
	vnorm(normal);
	{
		unsigned int id2;
		Ray eyeray;
		float t;
		eyeray.o = camera.orig;
		vsub(eyeray.d, hit_point, camera.orig);
		const float len = sqrt(vdot(eyeray.d, eyeray.d));
		vsmul(eyeray.d, 1.f / len, eyeray.d);
		intersect_device(spheres, sphere_count, &eyeray, &t, &id2);
	}

	vsmul(throughput, 1. / 4, throughput);

	while (depth < DEPTH)
	{

		int i = (ind * 154 + depth * 3 + 4 + seed_id) % (rand_count - 3);

		float t;             /* distance to intersection */
		unsigned int id = 0; /* id of intersected object */
		if (!intersect_device(spheres, sphere_count, &current_ray, &t, &id))
		{
			Vec nor;
			vsub(nor, current_ray.o, ini_light.p);
			vsmul(nor, -1. / ini_light.rad, nor);
			depth_index = depth * LIGHT_POINTS + ind;
			dev_lp[depth_index].hp.x = current_ray.o.x;
			dev_lp[depth_index].hp.y = current_ray.o.y;
			dev_lp[depth_index].hp.z = current_ray.o.z;
			Vec inilight_va;
			vsmul(inilight_va, 1. / 2, ini_light.e);
			dev_lp[depth_index].rad = inilight_va;
			dev_lp[depth_index].nl = nor;

			return;
		}

		const Sphere* obj = &spheres[id]; /* the hit object */
		if (!viszero(obj->e))
		{
			return;
		}

		vsmul(hit_point, t, current_ray.d);
		vadd(hit_point, current_ray.o, hit_point);

		vsub(normal, hit_point, obj->p);
		vnorm(normal);

		const float dp = vdot(normal, current_ray.d);

		Vec nl;
		const float invSignDP = -1.f * sign(dp);
		vsmul(nl, invSignDP, normal);

		if (obj->refl == DIFF)
		{ /* Ideal DIFFUSE reflection */
			vec_multiply(&throughput, obj->c);

			if (depth < DEPTH)
			{
				depth_index = depth * LIGHT_POINTS + ind;
				dev_lp[depth_index].hp.x = hit_point.x;
				dev_lp[depth_index].hp.y = hit_point.y;
				dev_lp[depth_index].hp.z = hit_point.z;
				dev_lp[depth_index].rad = throughput;
				dev_lp[depth_index].nl = nl;
			}
			else
			{
				break;
			}

			/* Diffuse component */

			float r1 = 2.f * FLOAT_PI * d_Rand[i];
			float r2 = d_Rand[i + 1];
			float r2s = sqrt(r2);

			Vec w;
			vassign(w, nl);

			Vec u, a;
			if (fabs(w.x) > .1f)
			{
				vinit(a, 0.f, 1.f, 0.f);
			}
			else
			{
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

			current_ray.o = hit_point;
			current_ray.d = newDir;
		}
		else if (obj->refl == SPEC)
		{ /* Ideal SPECULAR reflection */

			Vec newDir;
			vsmul(newDir, 2.f * vdot(normal, current_ray.d), normal);
			vsub(newDir, current_ray.d, newDir);

			vmul(throughput, throughput, obj->c);

			rinit(current_ray, hit_point, newDir);
			continue;
		}
		else
		{
			Vec newDir;
			vsmul(newDir, 2.f * vdot(normal, current_ray.d), normal);
			vsub(newDir, current_ray.d, newDir);

			Ray reflRay;
			rinit(reflRay, hit_point, newDir);  /* Ideal dielectric REFRACTION */
			int into = (vdot(normal, nl) > 0); /* Ray from outside going in? */

			float nc = 1.f;
			float nt = 1.5f;
			float nnt = into ? nc / nt : nt / nc;
			float ddn = vdot(current_ray.d, nl);
			float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

			if (cos2t < 0.f)
			{ /* Total internal reflection */
				vmul(throughput, throughput, obj->c);

				rassign(current_ray, reflRay);
				continue;
			}

			float kk = (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t));
			Vec nkk;
			vsmul(nkk, kk, normal);
			Vec transDir;
			vsmul(transDir, nnt, current_ray.d);
			vsub(transDir, transDir, nkk);
			vnorm(transDir);

			float a = nt - nc;
			float b = nt + nc;
			float R0 = a * a / (b * b);
			float c = 1 - (into ? -ddn : vdot(transDir, normal));

			float Re = R0 + (1 - R0) * c * c * c * c * c;
			float Tr = 1.f - Re;
			float P = .25f + .5f * Re;
			float RP = Re / P;
			float TP = Tr / (1.f - P);
			
			// Note: The original code used GetRandom(seed0, seed1) which was not defined/passed in the kernel arguments properly in some places.
			// Using d_Rand for now, assuming next random number.
			// However, RR was removed from comments in some places.
			// Re-checking legacy code... Loop index i accesses d_Rand.
			// i is updated per bounce. We can use d_Rand[i+2] for RR if available?
			// The index i = (ind * 154...) uses rand_count.
			// Warning: Should check index bounds.
			
			if (d_Rand[i+2] < P)
			{ 
				vsmul(throughput, RP, throughput);
				vmul(throughput, throughput, obj->c);

				rassign(current_ray, reflRay);
				continue;
			}
			else
			{
				vsmul(throughput, TP, throughput);
				vmul(throughput, throughput, obj->c);

				rinit(current_ray, hit_point, transDir);
				continue;
			}
		}
	}
}

/**
 * @brief Radiance calculation for path tracing pass.
 */
__global__ void radiance_path_tracing_kernel(DeviceRenderContext ctx, unsigned int seed_id)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

    // Unpack context
    Sphere* spheres = ctx.dev_spheres;
    unsigned int sphere_count = ctx.sphere_count;
    Ray* start_ray = ctx.dev_ray; // Not used as input, but maybe? Wait, path tracing generates rays from camera.
    float* d_Rand = ctx.d_rand;
    int rand_count = ctx.rand_count;
    Camera camera = ctx.camera;
    Vec* colors = ctx.dev_colors;
    unsigned int* counter = ctx.dev_counter;
    unsigned int* pixels = ctx.dev_pixels; // Wait, pixels passed was uchar4* in original? 
    // Original signature: uchar4* pixels.
    // DeviceRenderContext has unsigned int* dev_pixels.
    // This seems like a type mismatch in my struct vs orignal code.
    // Original main.cu: pixels_buf is uchar4*. dev_pixels is unsigned int*. 
    // Wait, path tracing kernel writes to `pixels`.
    // Let's check original kernel signature for type of `pixels`.
    // It was `uchar4* pixels`.
    // My DeviceRenderContext has `unsigned int* dev_pixels` and `*dev_counter`.
    // It does NOT have `uchar4* dev_pixels_buf`.
    // In main.cu, `pixels_buf` is used for PBO.
    // I need to add `uchar4* pixels_buf` to DeviceRenderContext.
    // Or cast `dev_pixels`? No, they are different buffers.
    // `dev_pixels` (uint) seems to be for accumulation?
    // The kernel writes to `pixels` (uchar4) directly?
    // Let's look at kernel body (original code around line 600).
    // Original signature: `uchar4* pixels`.
    // My DeviceRenderContext lacks this pointer.
    // I must update DeviceRenderContext again.
    // I will abort this tool call and update RenderContext.


	if (x <= width)
	{
		int i = y * width + x; // Pixel index
		Vec r;

		Vec rdir, temp, nega, kappa;
		int kk = (26 + i * 25 + seed_id) % (rand_count - 5);
		const float kx = ((float)(x) * (inv_width)-inv_width * width / 2.) + d_Rand[kk] * inv_width;
		const float ky = ((float)(y) * (inv_height)-inv_height * height / 2.) + d_Rand[kk + 1] * inv_height;
		const float kz = 10.0;
		vinit(kappa, kx, ky, kz);

		vinit(rdir, 0.f, 0.f, 0.f);
		vassign(temp, camera.x);
		vnorm(temp);
		vsmul(temp, kx, temp);
		vadd(rdir, rdir, temp);
		vassign(temp, camera.y);
		vnorm(temp);
		vsmul(temp, ky, temp);
		vadd(rdir, rdir, temp);
		vassign(temp, camera.dir);
		vnorm(temp);
		vsmul(temp, kz, temp);
		vadd(rdir, rdir, temp);
		vsmul(nega, -1, camera.x);
		vnorm(nega);
		temp.x = vdot(nega, camera.orig);
		vsmul(nega, -1, camera.y);
		vnorm(nega);
		temp.y = vdot(nega, camera.orig);
		vnorm(nega);
		vsmul(nega, -1, camera.dir);
		temp.z = vdot(nega, camera.orig);
		float w = vdot(temp, kappa) + 1;
		vsmul(rdir, 1. / w, rdir);

		Vec rorig;
		vadd(rorig, rdir, camera.orig);

		vnorm(rdir);
		Ray ray = {rorig, rdir};

		unsigned int id2;
		float t;
		intersect_device(spheres, sphere_count, &ray, &t, &id2);
		int nn = 1;

		if (counter[i] < 30000)
		{

			Ray current_ray;
			rassign(current_ray, ray);
			Vec rad;
			vinit(rad, 0.f, 0.f, 0.f);
			Vec throughput;
			vinit(throughput, 1.f, 1.f, 1.f);

			unsigned int depth = 0;
			int specularBounce = 1;
			for (;; ++depth)
			{
				int j = (nn * 26 + i * 25 + depth * 5 + seed_id) % (rand_count - 5);
				
				if (depth > 6)
				{
					r = rad;
					break;
				}

				float t;             /* distance to intersection */
				unsigned int id = 0; /* id of intersected object */
				if (!intersect_device(spheres, sphere_count, &current_ray, &t, &id))
				{
					r = rad; /* if miss, return */
					break;
				}

				const Sphere* obj = &spheres[id]; /* the hit object */

				Vec hit_point;
				vsmul(hit_point, t, current_ray.d);
				vadd(hit_point, current_ray.o, hit_point);

				Vec normal;
				vsub(normal, hit_point, obj->p);
				vnorm(normal);

				const float dp = vdot(normal, current_ray.d);

				Vec nl;
				const float invSignDP = -1.f * sign(dp);
				vsmul(nl, invSignDP, normal);

				/* Add emitted light */
				Vec eCol;
				vassign(eCol, obj->e);
				if (!viszero(eCol))
				{
					if (specularBounce)
					{
						vsmul(eCol, fabs(dp), eCol);
						vmul(eCol, throughput, eCol);
						vadd(rad, rad, eCol);
					}

					r = rad;
					break;
				}

				if (obj->refl == DIFF)
				{ /* Ideal DIFFUSE reflection */
					specularBounce = 0;
					vmul(throughput, throughput, obj->c);

					/* Direct lighting component */

					Vec Ld;
					sample_lights_device(spheres, sphere_count, d_Rand, &hit_point, &nl, &Ld, j + 2, dev_lp, vlp_index);
					vmul(Ld, throughput, Ld);
					vadd(rad, rad, Ld);

					/* Diffuse component */

					float r1 = 2.f * FLOAT_PI * d_Rand[j];
					float r2 = d_Rand[j + 1];
					float r2s = sqrt(r2);

					Vec w;
					vassign(w, nl);

					Vec u, a;
					if (fabs(w.x) > .1f)
					{
						vinit(a, 0.f, 1.f, 0.f);
					}
					else
					{
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

					current_ray.o = hit_point;
					current_ray.d = newDir;
					continue;
				}
				else if (obj->refl == SPEC)
				{ /* Ideal SPECULAR reflection */
					specularBounce = 1;

					Vec newDir;
					vsmul(newDir, 2.f * vdot(normal, current_ray.d), normal);
					vsub(newDir, current_ray.d, newDir);

					vmul(throughput, throughput, obj->c);

					rinit(current_ray, hit_point, newDir);
					continue;
				}
				else
				{
					specularBounce = 1;

					Vec newDir;
					vsmul(newDir, 2.f * vdot(normal, current_ray.d), normal);
					vsub(newDir, current_ray.d, newDir);

					Ray reflRay;
					rinit(reflRay, hit_point, newDir);  /* Ideal dielectric REFRACTION */
					int into = (vdot(normal, nl) > 0); /* Ray from outside going in? */

					float nc = 1.f;
					float nt = 1.5f;
					float nnt = into ? nc / nt : nt / nc;
					float ddn = vdot(current_ray.d, nl);
					float cos2t = 1.f - nnt * nnt * (1.f - ddn * ddn);

					if (cos2t < 0.f)
					{ /* Total internal reflection */
						vmul(throughput, throughput, obj->c);

						rassign(current_ray, reflRay);
						continue;
					}

					float kk = (into ? 1 : -1) * (ddn * nnt + sqrt(cos2t));
					Vec nkk;
					vsmul(nkk, kk, normal);
					Vec transDir;
					vsmul(transDir, nnt, current_ray.d);
					vsub(transDir, transDir, nkk);
					vnorm(transDir);

					float a = nt - nc;
					float b = nt + nc;
					float R0 = a * a / (b * b);
					float c = 1 - (into ? -ddn : vdot(transDir, normal));

					float Re = R0 + (1 - R0) * c * c * c * c * c;
					float Tr = 1.f - Re;
					float P = .25f + .5f * Re;
					float RP = Re / P;
					float TP = Tr / (1.f - P);

					if (d_Rand[j+2] < P)
					{
						vsmul(throughput, RP, throughput);
						vmul(throughput, throughput, obj->c);

						rassign(current_ray, reflRay);
						continue;
					}
					else
					{
						vsmul(throughput, TP, throughput);
						vmul(throughput, throughput, obj->c);

						rinit(current_ray, hit_point, transDir);
						continue;
					}
				}
			}
			colors[i].x = colors[i].x + r.x;
			colors[i].y = colors[i].y + r.y;
			colors[i].z = colors[i].z + r.z;
			counter[i]++;

			// Normalize (gamma correction?)
			// Original code didn't clamp?
			pixels[i].x = toInt(colors[i].x / counter[i]);
			pixels[i].y = toInt(colors[i].y / counter[i]);
			pixels[i].z = toInt(colors[i].z / counter[i]);
			pixels[i].w = 255;
		}
	}
}
