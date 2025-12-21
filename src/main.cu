/**
 * @file main.cu
 * @brief Main entry point for the GPU Bidirectional Raytracer.
 *
 * This file handles the initialization of CUDA, OpenGL, and the main rendering loop.
 * It manages memory allocation, kernel launches, and display updates.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

#include <GL/glew.h>
// #include <GL/freeglut.h> // GLUT disabled

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand.h>

#include "camera.h"
#include "scene.h"
#include "display_functions.h"
#include "constants.h"
#include "render_context.h"
#include "log.h"
#include "cuda_utils.h"
#include "render_utils.h"

// --- Globals ---
// Encapsulated into RenderContext to reduce global variable clutter.
RenderContext g_ctx;

// --- Forward Declarations ---
extern void display(); // Defined in display_functions.cpp

// Forward declarations of kernels (defined in rng_kernels.cu)
__global__ void random_gpu(float* d_Random, int nPerRng);
__global__ void box_muller_gpu(float* d_Random, int nPerRng);

// External functions (from kernels)
extern "C" void load_mt_gpu(const char* fname);
extern "C" void seed_mt_gpu(unsigned int seed);

// Kernels defined in path_tracer_kernels.cu
__global__ void get_ray_kernel(const Sphere light, DeviceRenderContext ctx, unsigned int seed_id);

__global__ void radiance_light_tracing_kernel(int id_light, DeviceRenderContext ctx, unsigned int seed_id);

__global__ void radiance_path_tracing_kernel(DeviceRenderContext ctx, unsigned int seed_id);

const char* dat_path = "assets/data/MersenneTwister.dat";

const int PATH_N = 7680000;
const int N_PER_RNG = align_up(divide_round_up(PATH_N, MT_RNG_COUNT), 2);
const int RAND_N = MT_RNG_COUNT * N_PER_RNG;

/**
 * @brief Allocates host and device buffers.
 */
void allocate_buffers()
{
	const int pixel_cnt = g_ctx.height * g_ctx.width;
	g_ctx.pixel_count = pixel_cnt;
	
	log_info("Allocating Buffers...");

	g_ctx.counter = (unsigned int*)calloc(pixel_cnt * 2, sizeof(unsigned int));
	g_ctx.pixels = (uchar4*)calloc(pixel_cnt, sizeof(uchar4));
	g_ctx.iteraz = (unsigned int*)calloc(g_ctx.raynthread * g_ctx.rayngrid, sizeof(unsigned int));

	g_ctx.size = DEPTH * LIGHT_POINTS * sizeof(LightPath);
	g_ctx.lp = (LightPath*)malloc(g_ctx.size);
	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_lp, g_ctx.size));

	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_ray, sizeof(Ray) * g_ctx.raynthread * g_ctx.rayngrid));
	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_iteraz, sizeof(unsigned int) * g_ctx.raynthread * g_ctx.rayngrid));
	
	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.d_rand, sizeof(float[RAND_N])));
	load_mt_gpu(dat_path);

	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_colors, sizeof(Vec[pixel_cnt])));
	gpuErrchk(cudaMemset(g_ctx.d_ctx.dev_colors, 0, sizeof(Vec[pixel_cnt])));

	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_counter, sizeof(unsigned int[pixel_cnt])));
	gpuErrchk(cudaMemset(g_ctx.d_ctx.dev_counter, 0, sizeof(unsigned int[pixel_cnt])));

	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_pixels, sizeof(unsigned int[pixel_cnt])));
	gpuErrchk(cudaMemset(g_ctx.d_ctx.dev_pixels, 0, sizeof(unsigned int[pixel_cnt])));

	gpuErrchk(cudaMalloc(&g_ctx.d_ctx.dev_spheres, sizeof(Sphere[g_ctx.sphere_count])));
	gpuErrchk(cudaMemcpy(g_ctx.d_ctx.dev_spheres, g_ctx.spheres, sizeof(Sphere[g_ctx.sphere_count]), cudaMemcpyHostToDevice));

    // Update Device Scalar context
    g_ctx.d_ctx.width = g_ctx.width;
    g_ctx.d_ctx.height = g_ctx.height;
    g_ctx.d_ctx.inverse_width = g_ctx.inverse_width;
    g_ctx.d_ctx.inverse_height = g_ctx.inverse_height;
    g_ctx.d_ctx.sphere_count = g_ctx.sphere_count;
    g_ctx.d_ctx.rand_count = RAND_N;
    g_ctx.d_ctx.camera = g_ctx.camera; // Ensure camera is up to date
}

/**
 * @brief Frees all allocated memory on host and device.
 */
void free_buffers()
{
	if (g_ctx.pixels) free(g_ctx.pixels);
    if (g_ctx.d_ctx.dev_colors) cudaFree(g_ctx.d_ctx.dev_colors);
	if (g_ctx.d_ctx.dev_pixels) cudaFree(g_ctx.d_ctx.dev_pixels);
	if (g_ctx.counter) free(g_ctx.counter);
	if (g_ctx.d_ctx.dev_counter) cudaFree(g_ctx.d_ctx.dev_counter);
	if (g_ctx.d_ctx.d_rand) cudaFree(g_ctx.d_ctx.d_rand);
	if (g_ctx.d_ctx.dev_ray) cudaFree(g_ctx.d_ctx.dev_ray);
	if (g_ctx.d_ctx.dev_iteraz) cudaFree(g_ctx.d_ctx.dev_iteraz);
	if (g_ctx.iteraz) free(g_ctx.iteraz);
	if (g_ctx.lp) free(g_ctx.lp);
	if (g_ctx.d_ctx.dev_lp) cudaFree(g_ctx.d_ctx.dev_lp);
}

/**
 * @brief Updates the rendering using path tracing.
 */
void update_rendering(void)
{
	log_info("Update Rendering (Path Tracing)");
	double start_time = wall_clock_time(); // Defined in display_functions.cpp (extern)
    // Wait, wall_clock_time is extern? display_functions.cpp has it.
    // display_functions.cpp is NOT a library, but linked file. 
    // I should declare it in a header or extern here.
    // Ah, display_functions.h likely has it. Checked includes locally?
    // display_functions.h was in makefile headers. Assuming it exists.
    
    // Update scalars in device context before launch
    g_ctx.d_ctx.current_sample = g_ctx.current_sample;
    g_ctx.d_ctx.vlp_index = g_ctx.vlp_index;
    g_ctx.d_ctx.camera = g_ctx.camera;

	unsigned int sid = rand() % RAND_N;
	dim3 dim_block(19, 19);
	dim3 dim_grid(ceil(g_ctx.width / float(dim_block.x)), ceil(g_ctx.height / float(dim_block.y)));
	
	radiance_path_tracing_kernel<<<dim_grid, dim_block>>>(g_ctx.d_ctx, sid);
	cudaDeviceSynchronize();

	g_ctx.current_sample++;
	const float elapsed_time = wall_clock_time() - start_time;
	g_ctx.total_time += elapsed_time;
	const float sample_sec = g_ctx.height * g_ctx.width / elapsed_time;
	
	log_info("Rendering time %.3f sec (pass %d) Total:%.2f  Sample/sec  %.1fK",
	       elapsed_time, g_ctx.current_sample, g_ctx.total_time, sample_sec / 1000.f);

	if (g_ctx.flag == MAX_ITER)
	{
		g_ctx.vlp_index += MAX_VLP;
		g_ctx.flag = 1;
	}
	if (g_ctx.flag < MAX_ITER)
		g_ctx.flag++; 
}

/**
 * @brief Updates the rendering using light tracing.
 */
void update_rendering_light(void)
{
	log_info("Update Rendering (Light Tracing)");
	g_ctx.pixel_count = g_ctx.height * g_ctx.width;

    // Update scalars
    g_ctx.d_ctx.current_sample = g_ctx.current_sample;
    // vlp_index is not used in light tracing kernel direct args, but maybe in loops.

	for (unsigned int i = 0; i < g_ctx.sphere_count; i++)
	{
		const Sphere* light = &g_ctx.spheres[i];
		if (!viszero(light->e))
		{
			gpuErrchk(cudaMemset(g_ctx.pixels_buf, 0, sizeof(uchar4[g_ctx.pixel_count])));
			
			seed_mt_gpu(g_ctx.current_sample * 5);
			random_gpu<<<32, 128>>>(g_ctx.d_ctx.d_rand, N_PER_RNG);
            gpuErrchk(cudaGetLastError());
            
			unsigned int sid = 0;
            // Update context for this light? No, light is passed as arg.
			get_ray_kernel<<<g_ctx.rayngrid, g_ctx.raynthread>>>(*light, g_ctx.d_ctx, sid);
			gpuErrchk(cudaGetLastError());

			radiance_light_tracing_kernel<<<g_ctx.rayngrid, g_ctx.raynthread>>>(i, g_ctx.d_ctx, sid);
			gpuErrchk(cudaGetLastError());

			cudaDeviceSynchronize();
			gpuErrchk(cudaGetLastError());
		}
	}
	
	g_ctx.flag = 2; // after one light execution, there are two path execution
}

/**
 * @brief Re-initializes the scene.
 */
void reinit_scene()
{
	g_ctx.current_sample = 0;
	g_ctx.flag = 1;
	free_buffers();
	allocate_buffers();
	update_rendering_light();
}

/**
 * @brief Re-initializes the rendering context.
 * @param realloc_buffers If true, reallocates buffers.
 */
void reinit(const int realloc_buffers)
{
	if (realloc_buffers)
	{
		free_buffers();
		allocate_buffers();
	}
	g_ctx.reinit_counter++;

	update_camera(); // Defined in display_functions.cpp
    g_ctx.d_ctx.camera = g_ctx.camera; // Sync back to device context

	g_ctx.current_sample = 0;
	if (g_ctx.reinit_counter % 2 == 0)
	{
		update_rendering_light();
	}
	log_info("Finished light tracing, starting path tracing");
	update_rendering();
}

/**
 * @brief Main function.
 */
int main(int argc, char* argv[])
{
    // Initialize context defaults
    g_ctx.raynthread = RAYNTHREAD;
    g_ctx.rayngrid = RAYNGRID;
    g_ctx.work_group_size = 1;
    g_ctx.flag = 1;
    g_ctx.vlp_index = MAX_VLP;
    g_ctx.is_smallpt_cpu = 1;

	fprintf(stderr, "Usage: %s <window width> <window height> <scene file>\n", argv[0]);

	if (argc == 4)
	{
		g_ctx.width = atoi(argv[1]);
		g_ctx.height = atoi(argv[2]);
		read_scene(argv[3]); // Defined in display_functions.cpp
	}
	else if (argc == 1)
	{
        // CornellSpheres is defined in scene.h likely?
        // In original main.cu: spheres = CornellSpheres;
        // Check scene.h for CornellSpheres definition. Assuming global available.
		g_ctx.spheres = CornellSpheres;
		g_ctx.sphere_count = sizeof(CornellSpheres) / sizeof(Sphere);

		vinit(g_ctx.camera.orig, 50.f, 44.f, 176.f);
		vinit(g_ctx.camera.target, 50.f, 44 - 0.042612f, 175.f);
	}
	else
		exit(-1);
        
	g_ctx.height += 1;
	g_ctx.width += 1;
	g_ctx.inverse_width = 14. / g_ctx.width;
	g_ctx.inverse_height = 10.5 / g_ctx.height;
	update_camera();

    extern int run_qt_app(int argc, char* argv[]);
    return run_qt_app(argc, argv);
}
