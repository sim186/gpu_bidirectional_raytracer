#ifndef RENDER_UTILS_H
#define RENDER_UTILS_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * @brief Creates a Pixel Buffer Object (PBO) for OpenGL-CUDA interop.
 * @param pbo Pointer to the PBO handle.
 * @param width Width of the buffer.
 * @param height Height of the buffer.
 */
extern "C" void create_pbo(GLuint* pbo, int width, int height);

/**
 * @brief Creates an OpenGL texture.
 * @param texture_id Pointer to the texture ID.
 * @param width Width of the texture.
 * @param height Height of the texture.
 */
extern "C" void create_texture(GLuint* texture_id, int width, int height);

/**
 * @brief Saves the current render to a PPM file.
 * @param pixels Device pointer to pixel data.
 * @param width Image width.
 * @param height Image height.
 * @param current_sample Current sample count (for filename).
 * @param total_time Total rendering time (for filename).
 * @param max_vlp Max VLP parameter (for filename).
 */
extern "C" void save_ppm(uchar4* pixels, int width, int height, int current_sample, float total_time, int max_vlp);

#endif // RENDER_UTILS_H
