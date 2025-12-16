#pragma once

#include <math.h>

// Jens's patch for MacOS
#ifdef __APPLE__
#include <GLut/glut.h>
#else
#include <GL/glut.h>
#endif

#include "vector_math.h"

extern int width;
extern float inverse_width;
extern int height;
extern float inverse_height;
extern uchar4* pixels;
extern unsigned int* counter;
extern unsigned int renderingFlags;
extern char caption_buffer[256];

extern int is_smallpt_cpu;

extern void InitGlut(int argc, char* argv[], char* window_title);
extern double WallClockTime();

extern void ReadScene(char*);
extern void UpdateCamera();
