#pragma once

#include <math.h>
#include <vector_types.h>

#include "vector_math.h"

extern int width;
extern float inverse_width;
extern int height;
extern float inverse_height;
extern uchar4* pixels;
extern unsigned int* counter;
extern char caption_buffer[256];

extern int is_smallpt_cpu;

// Function declarations
void init_glut(int argc, char* argv[], char* window_title);
double wall_clock_time();

void read_scene(char* file_name);
void update_camera();
