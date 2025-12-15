#
# smallptGPU & smallptCPU Makefile
#

CC=nvcc

CCFLAGS=-g -I$(ATISTREAMSDKROOT)/inc -Iinclude
#-v

LIB = -lm -lGLU -lGL -lglut -lGLEW -lcurand --ptxas-options=-v

SRC_DIR = src
INC_DIR = include

SOURCES = $(SRC_DIR)/device.cu $(SRC_DIR)/smallpt_cpu.c $(SRC_DIR)/display_func.c $(SRC_DIR)/MersenneTwister_kernel.cu
HEADERS = $(INC_DIR)/vec.h $(INC_DIR)/camera.h $(INC_DIR)/geom.h $(INC_DIR)/display_func.h $(INC_DIR)/simplernd.h $(INC_DIR)/scene.h $(INC_DIR)/geomfunc.h

smallptCPU: $(SOURCES) $(HEADERS)
	$(CC) $(CCFLAGS) -DSMALLPT_CPU -o smallptCPU $(SOURCES) $(LIB)
clean:
	rm -rf smallptCPU smallptGPU image.ppm SmallptGPU-v1.6 smallptgpu-v1.6.tgz preprocessed_rendering_kernel.cl

cleanobj:
	rm -f core *.o a.out
.f.o:
	$(FC) $(FFLAGS) -c $*.f
.c.o:
	$(CC) $(INC) $(CFLAGS) -c $*.c

