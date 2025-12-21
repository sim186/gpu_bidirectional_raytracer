/**
 * @file display_functions.cpp
 * @brief Helper functions for scene loading and camera updates.
 *        Refactored to remove GLUT dependencies and use RenderContext.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
#include <math.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/time.h>
#include <unistd.h>
#elif defined(WIN32)
#include <windows.h>
#else
// Unsupported Platform
#endif

#include "camera.h"
#include "geometry.h"
#include "display_functions.h"
#include "render_context.h"
#include "log.h"

// Externs
extern RenderContext g_ctx;

/**
 * @brief Returns the current wall clock time in seconds.
 */
double wall_clock_time()
{
 #if defined(__linux__) || defined(__APPLE__)
	struct timeval t;
	gettimeofday(&t, NULL);
	return t.tv_sec + t.tv_usec / 1000000.0;
 #elif defined(WIN32)
	return GetTickCount() / 1000.0;
 #else
	return 0.0;
 #endif
}

/**
 * @brief Reads the scene definition from a file.
 * @param file_name Path to the scene file.
 */
void read_scene(char* file_name)
{
	log_info("Reading scene: %s", file_name);

	FILE* f = fopen(file_name, "r");
	if (!f)
	{
		log_error("Failed to open file: %s", file_name);
		exit(-1);
	}

	// Read camera
	if (fscanf(f, "camera %f %f %f  %f %f %f\n",
	               &g_ctx.camera.orig.x, &g_ctx.camera.orig.y, &g_ctx.camera.orig.z,
	               &g_ctx.camera.target.x, &g_ctx.camera.target.y, &g_ctx.camera.target.z) != 6)
	{
		log_error("Failed to read camera parameters");
		exit(-1);
	}

	// Read sphere count
	if (fscanf(f, "size %u\n", &g_ctx.sphere_count) != 1)
	{
		log_error("Failed to read sphere count");
		exit(-1);
	}
	log_info("Scene size: %d", g_ctx.sphere_count);

	// Read spheres
	g_ctx.spheres = (Sphere*)malloc(sizeof(Sphere) * g_ctx.sphere_count);
	for (unsigned int i = 0; i < g_ctx.sphere_count; i++)
	{
		Sphere* s = &g_ctx.spheres[i];
		int mat;
		if (fscanf(f, "sphere %f  %f %f %f  %f %f %f  %f %f %f  %d\n",
		               &s->rad,
		               &s->p.x, &s->p.y, &s->p.z,
		               &s->e.x, &s->e.y, &s->e.z,
		               &s->c.x, &s->c.y, &s->c.z,
		               &mat) != 11)
		{
			log_error("Failed to read sphere #%d", i);
			exit(-1);
		}
		
		switch (mat)
		{
		case 0: s->refl = DIFF; break;
		case 1: s->refl = SPEC; break;
		case 2: s->refl = REFR; break;
		case 3: s->refl = LITE; break;
		default:
			log_error("Unknown material type: %d", mat);
			exit(-1);
		}
	}

	fclose(f);
}

/**
 * @brief Updates the camera coordinate system based on position and target.
 */
void update_camera()
{
	vsub(g_ctx.camera.dir, g_ctx.camera.target, g_ctx.camera.orig);
	vnorm(g_ctx.camera.dir);

	const Vec up = {0.f, 1.f, 0.f};
	const float fov = (M_PI / 180.f) * 45.f;
	vxcross(g_ctx.camera.x, g_ctx.camera.dir, up);
	vnorm(g_ctx.camera.x);
	vsmul(g_ctx.camera.x, g_ctx.width * fov / g_ctx.height, g_ctx.camera.x);

	vxcross(g_ctx.camera.y, g_ctx.camera.x, g_ctx.camera.dir);
	vnorm(g_ctx.camera.y);
	vsmul(g_ctx.camera.y, fov, g_ctx.camera.y);
}
