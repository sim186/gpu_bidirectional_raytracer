#include "render_utils.h"
#include "cuda_utils.h"
#include "log.h"

extern "C" void create_pbo(GLuint* pbo, int width, int height)
{
	if (pbo)
	{
		int num_texels = width * height;
		int num_values = num_texels * 4;
		int size_tex_data = sizeof(GLubyte) * num_values;

		glGenBuffers(1, pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, *pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
		
		cudaError_t err = cudaGLRegisterBufferObject(*pbo);
		if (err != cudaSuccess) {
			log_error("CUDA GL Register Buffer failed: %s", cudaGetErrorString(err));
		}
	}
}

extern "C" void create_texture(GLuint* texture_id, int width, int height)
{
	glEnable(GL_TEXTURE_2D);
	glGenTextures(1, texture_id);
	glBindTexture(GL_TEXTURE_2D, *texture_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

extern "C" void save_ppm(uchar4* pixels, int width, int height, int current_sample, float total_time, int max_vlp)
{
	char name[64];
    // Note: We need a host buffer to copy to. 
    // In the original code, 'pixels' was a host pointer that was memcpy'd from 'pixels_buf' (device).
    // Here we assume 'pixels' passed is the HOST buffer containing data.
    // If the logical flow in main.cu is Memcpy D->H then save, this function should take the HOST buffer.
    
	sprintf(name, "max%d_seconds%.3f_exe%d.ppm", max_vlp, total_time, current_sample);

	FILE* f = fopen(name, "w");
	if (!f)
	{
		log_error("Failed to open image file: %s", name);
	}
	else
	{
		fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
		for (int y = height - 1; y >= 0; --y)
		{
			unsigned char* p = (unsigned char*)(&pixels[y * width]);
			for (int x = 0; x < width; ++x, p += 4)
				fprintf(f, "%d %d %d ", p[0], p[1], p[2]);
		}
		fclose(f);
        log_info("Saved image to %s", name);
	}
}
