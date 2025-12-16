/* Renamed to snake_case filename and updated local variable names to snake_case
   This file is a mostly line-for-line copy of `smallptCPU.c` with locals renamed:
   - currentSample -> current_sample
   - TotalTime     -> total_time
   - pixelCount    -> pixel_count
   - ReInitCounter -> reinit_counter
   Also updated include to `display_func.h` (the new snake_case header).
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <rendercheck_gl.h>
#include "camera.h"
#include "scene.h"
#include "display_functions.h"
#include <curand.h>
#include "constants.h"

int raynthread = RAYNTHREAD;
int rayngrid = RAYNGRID;
#define MT_RNG_COUNT 4096

int workGroupSize = 1;
static int current_sample = 0;
static float total_time = 0;
Vec* dev_colors;
Ray* dev_ray;
Camera camera;
Sphere *spheres, *dev_spheres;
unsigned int sphereCount, *dev_counter, *dev_pixels, *dev_iteraz, *iteraz;
uchar4* pixels_buf;
GLuint pbo = NULL;
GLuint textureID = NULL;
int allFlag = 0;
int reinit_counter;
LightPath *lp, *dev_lp;
int vlp_index = MAX_VLP;
extern void display();
const char* dat_path = "assets/data/MersenneTwister.dat";
float* d_Rand;
extern int flag;
size_t size;
int pixel_count;

/* Divide a by b, rounding up to the nearest integer */
extern "C" int DivideRoundUp(int a, int b)
{
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

/* Align a to the nearest higher multiple of b */
extern "C" int AlignUp(int a, int b)
{
	return ((a % b) != 0) ? (a - a % b + b) : a;
}

const int PATH_N = 7680000;
const int N_PER_RNG = AlignUp(DivideRoundUp(PATH_N, MT_RNG_COUNT), 2);
const int RAND_N = MT_RNG_COUNT * N_PER_RNG;

__global__ void RandomGPU(float* d_Random, int nPerRng);
__global__ void BoxMullerGPU(float* d_Random, int nPerRng);
void loadMTGPU(const char* fname);
void seedMTGPU(unsigned int seed);
void UpdateRendering(void);

__global__ void GetRayKernel(const Sphere light, Ray* dev_data,
                             float* d_Rand, int current_sample,
                             int pixel_count, unsigned int sid);

__global__ void RadianceLightTracingKernel(Sphere* spheres, unsigned int sphereCount,
                                           Ray* startRay, float* d_Rand, int light_id,
                                           int pixel_count, Camera camera, Vec* colors,
                                           unsigned int* counter, LightPath* dev_lp,
                                           float inverse_width, float inverse_height,
                                           float width, float height, unsigned int sid);

__global__ void RadiancePathTracingKernel(Sphere* spheres, unsigned int sphereCount,
                                          float* d_Rand, int pixel_count, Camera camera,
                                          Vec* colors, unsigned int* counter,
                                          uchar4* pixels, float inverse_width,
                                          float inverse_height, float width, float height,
                                          unsigned int sid, unsigned int param,
                                          LightPath* dev_lp, int vlp_index);

void FreeBuffers()
{
	free(pixels);
	cudaFree(dev_colors);
	cudaFree(dev_pixels);
	free(counter);
	cudaFree(dev_counter);
	cudaFree(d_Rand);
	cudaFree(dev_ray);
	cudaFree(dev_iteraz);
	free(iteraz);
	free(lp);
	cudaFree(dev_lp);
}

void CreatePBO(GLuint* pbo)
{
	if (pbo)
	{
		int num_texels = width * height;
		int num_values = num_texels * 4;
		int size_tex_data = sizeof(GLubyte) * num_values;

		glGenBuffers(1, pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
		cudaGLRegisterBufferObject(*pbo);
	}
}

void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y)
{
	// Enable Texturing
	glEnable(GL_TEXTURE_2D);

	// Generate a texture identifier
	glGenTextures(1, textureID);

	// Make this the current texture (remember that GL is state-based)
	glBindTexture(GL_TEXTURE_2D, *textureID);

	// Allocate the texture memory. The last parameter is NULL since we only
	// want to allocate memory, not initialize it
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
	             GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// Must set the filter mode, GL_LINEAR enables interpolation when scaling
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	// Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
	// GL_TEXTURE_2D for improved performance if linear interpolation is
	// not desired. Replace GL_LINEAR with GL_NEAREST in the
	// glTexParameteri() call
}

void AllocateBuffers()
{
	const int pixel_count = height * width;
	cudaError_t error;

	fprintf(stderr, "Allocate Buffers\n");

	counter = (unsigned int*)calloc(pixel_count * 2, sizeof(unsigned int));
	pixels = (uchar4*)calloc(pixel_count, sizeof(uchar4));
	iteraz = (unsigned int*)calloc(raynthread * rayngrid, sizeof(unsigned int));

	size = DEPTH * LIGHT_POINTS * sizeof(LightPath);
	lp = (LightPath*)malloc(size);
	cudaMalloc(&dev_lp, size);

	error = cudaMalloc(&dev_ray, sizeof(Ray) * raynthread * rayngrid);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data dev_ray: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_iteraz, sizeof(unsigned int) * raynthread * rayngrid);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data dev_iteraz: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMalloc(&d_Rand, sizeof(float[RAND_N]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data d_Rand: %s\n",
		        cudaGetErrorString(error));
	}
	loadMTGPU(dat_path);

	error = cudaMalloc(&dev_colors, sizeof(Vec[pixel_count]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data dev_colors: %s\n",
		        cudaGetErrorString(error));
	}
	error = cudaMemset(&dev_colors, 0, sizeof(Vec[pixel_count]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to clean GPU data dev_colors: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_counter, sizeof(unsigned int[pixel_count]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data dev_counter: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMemset(dev_counter, 0, sizeof(unsigned int[pixel_count]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to clean GPU data dev_counter: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_pixels, sizeof(unsigned int[pixel_count]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data dev_pixels: %s\n",
		        cudaGetErrorString(error));
	}
	error = cudaMemset(&dev_pixels, 0, sizeof(unsigned int[pixel_count]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to clean GPU data dev_pixels: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_spheres, sizeof(Sphere[sphereCount]));
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to allocate GPU data dev_spheres: %s\n",
		        cudaGetErrorString(error));
	}

	error = cudaMemcpy(dev_spheres, spheres, sizeof(Sphere[sphereCount]), cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to upload GPU data dev_spheres: %s\n",
		        cudaGetErrorString(error));
	}
}

void SavePPM(int numbe)
{
	char name[32];
	cudaError_t error = cudaMemcpy(pixels, pixels_buf, sizeof(uchar4[pixel_count]), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to download GPU data pixels: %s\n", cudaGetErrorString(error));
	}
	sprintf(name, "max%d_secondi%.3f_exe%d.ppm", MAX_VLP, total_time, current_sample);

	FILE* f = fopen(name, "w"); // Write image to PPM file.
	if (!f)
	{
		fprintf(stderr, "Failed to open image file: image.ppm\n");
	}
	else
	{
		fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);

		int x, y;
		for (y = height - 1; y >= 0; --y)
		{
			unsigned char* p = (unsigned char*)(&pixels[y * width]);
			for (x = 0; x < width; ++x, p += 4)
				fprintf(f, "%d %d %d ", p[0], p[1], p[2]);
		}

		fclose(f);
	}
}

void UpdateRendering(void)
{
	printf("UpdateRendering\n");
	double startTime = WallClockTime();

	unsigned int sid = rand() % RAND_N;
	dim3 dimBlock(19, 19);
	dim3 dimGrid(ceil(width / float(dimBlock.x)), ceil(height / float(dimBlock.y)));
	RadiancePathTracingKernel<<<dimGrid, dimBlock>>>(dev_spheres, sphereCount,
	                                                 d_Rand, RAND_N, camera, dev_colors,
	                                                 dev_counter, pixels_buf, inverse_width,
	                                                 inverse_height, width, height, sid,
	                                                 30000, dev_lp, vlp_index);
	cudaThreadSynchronize();

	current_sample++;
	const float elapsedTime = WallClockTime() - startTime;
	total_time += elapsedTime;
	const float sampleSec = height * width / elapsedTime;
	// elapsedTime, currentSample, sampleSec / 1000.f);
	printf("Rendering time %.3f sec (pass %d) Total:%.2f  Sample/sec  %.1fK\n",
	       elapsedTime, current_sample, total_time, sampleSec / 1000.f);

	// else flag=1;               //altrimenti rilancio il light

	if (flag == MAX_ITER)
	{
		vlp_index += MAX_VLP;
		flag = 1;
	}
	if (flag < MAX_ITER)
		flag++; // in questo caso lancio 2 volte il path tracing
}

void UpdateRendering2(void)
{
	printf("UpdateRendering2\n");
	pixel_count = height * width;

	double startTime = WallClockTime();
	int i;

	// dim3 dimBlock(width*height);
	// dim3 dimGrid(ceil(width/float(dimBlock.x)),ceil(height/float(dimBlock.y)));

	for (i = 0; i < sphereCount; i++)
	{
		const Sphere* light = &spheres[i];
		if (!viszero(light->e))
		{

			cudaError_t error = cudaMemset(pixels_buf, 0, sizeof(uchar4[pixel_count]));
			if (error != cudaSuccess)
			{
				fprintf(stderr, "Unable to clean GPU data dev_counter: %s\n",
				        cudaGetErrorString(error));
			}
			error = cudaGetLastError();
			seedMTGPU(current_sample * 5);
			RandomGPU<<<32, 128>>>(d_Rand, N_PER_RNG);

			error = cudaGetLastError();
			if (error != cudaSuccess)
			{
				fprintf(stderr, "Kernel RandomGPU failed: %s\n",
				        cudaGetErrorString(error));
			}

			unsigned int sid = 0;
			GetRayKernel<<<rayngrid, raynthread>>>(*light, dev_ray, d_Rand, current_sample, RAND_N, sid);
			error = cudaGetLastError();
			if (error != cudaSuccess)
			{
				fprintf(stderr, "Kernel GetRayKernel failed: %s\n",
				        cudaGetErrorString(error));
			}

			RadianceLightTracingKernel<<<rayngrid, raynthread>>>(dev_spheres, sphereCount, dev_ray, d_Rand, i,
			                                                     RAND_N, camera, dev_colors, dev_counter,
			                                                     dev_lp, inverse_width, inverse_height, width, height,
			                                                     sid);
			error = cudaGetLastError();
			if (error != cudaSuccess)
			{
				fprintf(stderr, "Kernel Light Tracing failed: %s\n",
				        cudaGetErrorString(error));
			}

			cudaThreadSynchronize();

			error = cudaGetLastError();
			if (error != cudaSuccess)
			{
				fprintf(stderr, "Kernel RadianceLightTracing failed: %s\n",
				        cudaGetErrorString(error));
			}
		}
	}
	// else flag=1;
	flag = 2; // dopo una esecuzione di light ci sono due esecuzioni di path
}

void ReInitScene()
{
	current_sample = 0;
	flag = 1;
	FreeBuffers();
	AllocateBuffers();
	UpdateRendering2();
}

void ReInit(const int reallocBuffers)
{
	// Check if I have to reallocate buffers
	if (reallocBuffers)
	{
		FreeBuffers();
		AllocateBuffers();
	}
	reinit_counter++;

	UpdateCamera();
	current_sample = 0;
	if (reinit_counter % 2 == 0)
	{
		UpdateRendering2();
	}
	printf("Ho finito il light e comincio il path\n");
	UpdateRendering();
}

int main(int argc, char* argv[])
{
	is_smallpt_cpu = 1;

	fprintf(stderr, "Usage: %s\n", argv[0]);
	fprintf(stderr, "Usage: %s <window width> <window height> <scene file>\n", argv[0]);

	if (argc == 4)
	{
		width = atoi(argv[1]);
		height = atoi(argv[2]);
		ReadScene(argv[3]);
	}
	else if (argc == 1)
	{
		spheres = CornellSpheres;
		sphereCount = sizeof(CornellSpheres) / sizeof(Sphere);

		vinit(camera.orig, 50.f, 44.f, 176.f);
		vinit(camera.target, 50.f, 44 - 0.042612f, 175.f);
	}
	else
		exit(-1);
	height += 1;
	width += 1;
	inverse_width = 14. / width;
	inverse_height = 10.5 / height;
	UpdateCamera();

	InitGlut(argc, argv, "DR");

	/*------------------------------------------------------------------------*/
	// initGL(argc,argv);

	// glewInit();

	cudaError_t error = cudaGLSetGLDevice(cutGetMaxGflopsDeviceId());
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Unable to set GL Device: %s\n",
		        cudaGetErrorString(error));
	}

	createPBO(&pbo);
	createTexture(&textureID, width, height);
	CUT_CHECK_ERROR_GL();

	AllocateBuffers();

	/*------------------------------------------------------------------------*/

	glutMainLoop();

	return 0;
}
