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
#include "displayfunc.h"
#include <curand.h>
#include "cons.h"

int raynthread=RAYNTHREAD;
int rayngrid=RAYNGRID;
#define  MT_RNG_COUNT 4096


int workGroupSize = 1;
static int currentSample = 0;
static float TotalTime = 0;
Vec *dev_colors;
Ray *dev_ray;
Camera camera;
Sphere *spheres, *dev_spheres;
unsigned int sphereCount, *dev_counter,  *dev_pixels, *dev_iteraz, *iteraz;
uchar4 *pixels_buf;
GLuint pbo=NULL;
GLuint textureID=NULL;
int allFlag=0;
int ReInitCounter;
LightPath *lp,*dev_lp;
int vlp_index=MAX_VLP;
extern void display();
const char *dat_path = "assets/data/MersenneTwister.dat";
float *d_Rand ;
extern int flag;
size_t size;
int pixelCount;




//ceil(a / b)
extern "C" int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
extern "C" int iAlignUp(int a, int b){
    return ((a % b) != 0) ?  (a - a % b + b) : a;
}

const int    PATH_N = 7680000;
const int N_PER_RNG = iAlignUp(iDivUp(PATH_N, MT_RNG_COUNT), 2);
const int    RAND_N = MT_RNG_COUNT * N_PER_RNG;



__global__ void RandomGPU(float *d_Random,int nPerRng);
__global__ void BoxMullerGPU(float *d_Random, int nPerRng);
void loadMTGPU(const char *fname);
void seedMTGPU(unsigned int seed);
void UpdateRendering(void);

__global__ void get_ray(const Sphere light, Ray *dev_data, 
                        float *d_Rand, int cS, 
                        int pC,unsigned int sid);
                        
__global__ void RadianceLightTracing_dev(Sphere *spheres, unsigned int sphereCount, 
                                         Ray *startRay, float *d_Rand, int idlight, 
                                         int pC, Camera camera, Vec *colors, 
                                         unsigned int *counter, LightPath *dev_lp, 
                                         float invWidth, float invHeight, 
                                         float width, float height,unsigned int sid);
                                         
__global__ void RadiancePathTracing_dev(Sphere *spheres, unsigned int sphereCount, 
                                        float *d_Rand, int pC, Camera camera, 
                                        Vec *colors,unsigned int *counter, 
                                        uchar4 *pixels,float invWidth, 
                                        float invHeight, float width, float height, 
                                        unsigned int sid, unsigned int param,
                                        LightPath *dev_lp, int vlp_index);

void FreeBuffers() {
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

void createPBO(GLuint* pbo)
{

  if (pbo) {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;
    
    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1,pbo);
    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
    cudaGLRegisterBufferObject( *pbo );
  }
}

void createTexture(GLuint* textureID, unsigned int size_x, unsigned int size_y)
{
  // Enable Texturing
  glEnable(GL_TEXTURE_2D);

  // Generate a texture identifier
  glGenTextures(1,textureID);

  // Make this the current texture (remember that GL is state-based)
  glBindTexture( GL_TEXTURE_2D, *textureID);

  // Allocate the texture memory. The last parameter is NULL since we only
  // want to allocate memory, not initialize it
  glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
 			GL_BGRA,GL_UNSIGNED_BYTE, NULL);

  // Must set the filter mode, GL_LINEAR enables interpolation when scaling
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
  // Note: GL_TEXTURE_RECTANGLE_ARB may be used instead of
  // GL_TEXTURE_2D for improved performance if linear interpolation is
  // not desired. Replace GL_LINEAR with GL_NEAREST in the
  // glTexParameteri() call


}

void AllocateBuffers() {
	const int pixelCount = height * width ;
	cudaError_t error;

	fprintf(stderr,"Allocate Buffers\n");

    counter = (unsigned int*) calloc(pixelCount*2,sizeof(unsigned int));
	pixels = (uchar4 *) calloc(pixelCount,sizeof(uchar4));
	iteraz = (unsigned int*) calloc(raynthread*rayngrid,sizeof(unsigned int));

	
	size=DEPTH*LIGHT_POINTS*sizeof(LightPath);
    lp = (LightPath *)malloc(size);
    cudaMalloc(&dev_lp,      size);
 
	error = cudaMalloc(&dev_ray, sizeof(Ray)*raynthread*rayngrid);
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data dev_ray: %s\n",
			cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_iteraz, sizeof(unsigned int)*raynthread*rayngrid);
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data dev_iteraz: %s\n",
			cudaGetErrorString(error));
	}

	error = cudaMalloc(&d_Rand, sizeof(float[RAND_N]));	
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data d_Rand: %s\n",
			cudaGetErrorString(error));
	}
	loadMTGPU(dat_path);

	error = cudaMalloc(&dev_colors, sizeof(Vec[pixelCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data dev_colors: %s\n",
			cudaGetErrorString(error));
	}
	error = cudaMemset(&dev_colors, 0, sizeof(Vec[pixelCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to clean GPU data dev_colors: %s\n",
			cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_counter, sizeof(unsigned int[pixelCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data dev_counter: %s\n",
			cudaGetErrorString(error));
	}

	error = cudaMemset(dev_counter, 0, sizeof(unsigned int[pixelCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to clean GPU data dev_counter: %s\n",
			cudaGetErrorString(error));
	}

	

	error = cudaMalloc(&dev_pixels, sizeof(unsigned int[pixelCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data dev_pixels: %s\n",
			cudaGetErrorString(error));
	}
	error = cudaMemset(&dev_pixels, 0, sizeof(unsigned int[pixelCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to clean GPU data dev_pixels: %s\n",
			cudaGetErrorString(error));
	}

	error = cudaMalloc(&dev_spheres, sizeof(Sphere[sphereCount]));
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to allocate GPU data dev_spheres: %s\n",
			cudaGetErrorString(error));
	}

	error = cudaMemcpy(dev_spheres, spheres, sizeof(Sphere[sphereCount]), cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to upload GPU data dev_spheres: %s\n",
			cudaGetErrorString(error));
	}



}

void savePPM(int numbe){
char name[32];
cudaError_t error = cudaMemcpy(pixels, pixels_buf, sizeof(uchar4[pixelCount]), cudaMemcpyDeviceToHost);
	    if (error != cudaSuccess) {
	        fprintf(stderr, "Unable to download GPU data pixels: %s\n",cudaGetErrorString(error));
	    }
sprintf(name,"max%d_secondi%.3f_exe%d.ppm",MAX_VLP,TotalTime,currentSample);

FILE *f = fopen(name, "w"); // Write image to PPM file.
if (!f) {
	fprintf(stderr, "Failed to open image file: image.ppm\n");
} else {
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);

	int x, y;
	for (y = height - 1; y >= 0; --y) {
		unsigned char *p = (unsigned char *)(&pixels[y * width]);
		for (x = 0; x < width; ++x, p += 4)
			fprintf(f, "%d %d %d ", p[0], p[1], p[2]);
	}

	fclose(f);
}
}


void UpdateRendering(void) {
    printf("UpdateRendering\n");
	double startTime = WallClockTime();
	

	unsigned int sid=rand()%RAND_N;
	dim3 dimBlock(19,19);
	dim3 dimGrid(ceil(width/float(dimBlock.x)),ceil(height/float(dimBlock.y)));
	RadiancePathTracing_dev<<<dimGrid,dimBlock>>>(dev_spheres,sphereCount,
	                                              d_Rand,RAND_N,camera,dev_colors,
	                                              dev_counter,pixels_buf,invWidth,
	                                              invHeight,width,height,sid,
	                                              30000,dev_lp,vlp_index);
	cudaThreadSynchronize();
	

    currentSample++;
	const float elapsedTime = WallClockTime() - startTime;
	TotalTime += elapsedTime;
	const float sampleSec = height * width / elapsedTime;
	//sprintf(captionBuffer, "Rendering time %.3f sec (pass %d)  Sample/sec  %.1fK\n",
		//elapsedTime, currentSample, sampleSec / 1000.f);
	printf("Rendering time %.3f sec (pass %d) Total:%.2f  Sample/sec  %.1fK\n",
		elapsedTime, currentSample, TotalTime, sampleSec / 1000.f);
		//if(currentSample==100) {savePPM(1); for(;;) printf("Loop\n");}

//if(flag==1 || flag==2) flag++; //in questo caso lancio 2 volte il path tracing
	//else flag=1;               //altrimenti rilancio il light
	
	
	if(flag == MAX_ITER) {vlp_index+=MAX_VLP; flag=1;}
    if(flag < MAX_ITER) flag++; //in questo caso lancio 2 volte il path tracing
    //flag=2;


	
}


void UpdateRendering2(void) {
    printf("UpdateRendering2\n");
	pixelCount = height * width ;

	double startTime = WallClockTime();
	int i;
	
	//dim3 dimBlock(width*height);
	//dim3 dimGrid(ceil(width/float(dimBlock.x)),ceil(height/float(dimBlock.y)));


	for (i = 0; i < sphereCount; i++) {
	const Sphere *light = &spheres[i];
		if (!viszero(light->e)) {
			
				cudaError_t error = cudaMemset(pixels_buf, 0, sizeof(uchar4[pixelCount]));
				if (error != cudaSuccess) {
					fprintf(stderr, "Unable to clean GPU data dev_counter: %s\n",
					cudaGetErrorString(error));
				}
			    //printf("CC:%d\n",currentSample);
				error = cudaGetLastError();
				seedMTGPU(currentSample*5);
				RandomGPU<<<32, 128>>>(d_Rand, N_PER_RNG);
			
				error = cudaGetLastError();
				if (error != cudaSuccess) {
					fprintf(stderr, "Kernel RandomGPU failed: %s\n",
					cudaGetErrorString(error));
				}
			
				unsigned int sid=0;
				get_ray<<<rayngrid,raynthread>>>(*light, dev_ray, d_Rand,currentSample,RAND_N,sid);
				error = cudaGetLastError();
				if (error != cudaSuccess) {
					fprintf(stderr, "Kernel get_ray failed: %s\n",
					cudaGetErrorString(error));
				}
	        
	       
	     
				RadianceLightTracing_dev<<<rayngrid,raynthread>>>(dev_spheres, sphereCount, dev_ray, d_Rand, i, 
			                                                  RAND_N, camera, dev_colors, dev_counter, 
			                                                  dev_lp, invWidth, invHeight, width, height, 
			                                                  sid);
				error = cudaGetLastError();
				if (error != cudaSuccess) {
					fprintf(stderr, "Kernel Light Tracing failed: %s\n",
					cudaGetErrorString(error));
				}
			
				cudaThreadSynchronize();
			
	         
				//cudaMemcpy(lp,dev_lp,size,cudaMemcpyDeviceToHost);
				//int k,j;
				//for(k=0;k<rayngrid*raynthread;k++){
				    //printf("x=%.3f, %.3f, %.3f\n",lp[k].rad.x,lp[k].rad.y,lp[k].rad.z);
			    //}
			    //printf("x=%.3f, %.3f, %.3f\n",lp[0].hp.x,lp[0].hp.y,lp[0].hp.z);
	        
	            //for(j = 0; j < LIGHT_POINTS; j++){
	                //for(k = 0; k < DEPTH; k++){
						//printf("LightPath=%d, rad[%d](%.3f,%.3f,%.3f)\n",j,k*LIGHT_POINTS+j,lp[k*LIGHT_POINTS+j].rad.x ,lp[k*LIGHT_POINTS+j].rad.y,lp[k*LIGHT_POINTS+j].rad.z);
					//}
				//}
	         
				error = cudaGetLastError();
				if (error != cudaSuccess) {
					fprintf(stderr, "Kernel RadianceLightTracing failed: %s\n",
					cudaGetErrorString(error));
				}

			
	    }
    }
	//if(flag==1 || flag==2) flag++;
	//else flag=1;
	flag=2; //dopo una esecuzione di light ci sono due esecuzioni di path
}


void ReInitScene() {
	currentSample = 0;
	flag=1;
	FreeBuffers();
	AllocateBuffers();
	UpdateRendering2();
}

void ReInit(const int reallocBuffers) {
	// Check if I have to reallocate buffers
	if (reallocBuffers) {
		FreeBuffers();
		AllocateBuffers();
	}
    ReInitCounter++;

	//printf("ReinitCounter:%d\n",ReInitCounter);

	UpdateCamera();
	currentSample = 0;
	if(ReInitCounter%2==0){UpdateRendering2();}
	printf("Ho finito il light e comincio il path\n");
	UpdateRendering();
}

int main(int argc, char *argv[]) {
	amiSmallptCPU = 1;


	fprintf(stderr, "Usage: %s\n", argv[0]);
	fprintf(stderr, "Usage: %s <window width> <window height> <scene file>\n", argv[0]);

	if (argc == 4) {
		width = atoi(argv[1]);
		height = atoi(argv[2]);
		ReadScene(argv[3]);
	} else if (argc == 1) {
		spheres = CornellSpheres;
		sphereCount = sizeof(CornellSpheres) / sizeof(Sphere);

		vinit(camera.orig, 50.f, 44.f, 176.f);
		vinit(camera.target, 50.f, 44 - 0.042612f, 175.f);

		
		//vinit(camera.orig, 50.f, 45.f, 205.6f);
		//vinit(camera.orig, 50.f, 45.f, 155.6f);
		//vinit(camera.target, 50.f, 45.f, 204.6f);
		//vinit(camera.target, 50.f, 45 - 0.042612f, 204.6);
		//vinit(camera.target, 50.f, 45.f - 0.042612f, 4.6);
	} else
		exit(-1);
	height+=1;
	width+=1;
	invWidth=14./width;
	invHeight=10.5/height;
	UpdateCamera();

	InitGlut(argc, argv, "DR");

	/*------------------------------------------------------------------------*/
	//initGL(argc,argv);

	//glewInit();

	cudaError_t error=cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );
	if (error != cudaSuccess) {
		fprintf(stderr, "Unable to set GL Device: %s\n",
			cudaGetErrorString(error));
	}

	createPBO(&pbo);
	createTexture(&textureID,width,height);
	CUT_CHECK_ERROR_GL();

	AllocateBuffers();

	/*------------------------------------------------------------------------*/



    glutMainLoop( );

	return 0;
}


