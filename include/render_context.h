#ifndef RENDER_CONTEXT_H
#define RENDER_CONTEXT_H

#include "vector_math.h"
#include "camera.h"
#include "scene.h"
#include "constants.h"
#include <cuda_runtime.h>
#include <GL/glew.h>

typedef struct {
    // Device pointers
    Vec* dev_colors;
    Ray* dev_ray;
    Sphere* dev_spheres;
    unsigned int* dev_counter;
    unsigned int* dev_pixels;
    unsigned int* dev_iteraz;
    LightPath* dev_lp;
    float* d_rand;
    
    // Scalar configuration (device copy)
    int width;
    int height;
    float inverse_width;
    float inverse_height;
    unsigned int sphere_count;
    int rand_count;
    int current_sample;
    int vlp_index;
    Camera camera;
} DeviceRenderContext;

typedef struct {
    // Configuration
    int width;
    int height;
    float inverse_width;
    float inverse_height;
    int raynthread;
    int rayngrid;
    int work_group_size;
    int is_smallpt_cpu;
    
    // State
    int current_sample;
    float total_time;
    int all_flag;
    int reinit_counter;
    int vlp_index;
    int flag;
    
    // Host Data
    Camera camera;
    Sphere* spheres;
    unsigned int sphere_count;
    unsigned int* iteraz;
    unsigned int* counter;
    uchar4* pixels;     // Host buffer for PPM
    uchar4* pixels_buf; // Device buffer for Interop (actually mapped from PBO usually, or alloc'd)
                        // In original main.cu: pixels_buf passed to kernels. pixels used for saving.
                        // Wait, pixels_buf isn't alloc'd in allocate_buffers in original main.cu?
                        // Ah, pixels_buf is likely the registered GL buffer pointer.

    LightPath* lp;
    size_t size;
    int pixel_count;
    
    // OpenGL
    GLuint pbo;
    GLuint texture_id;
    
    // Device Context (copy of pointers for kernel/alloc referencing)
    DeviceRenderContext d_ctx;

} RenderContext;

#endif // RENDER_CONTEXT_H
