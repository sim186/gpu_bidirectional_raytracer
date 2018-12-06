#
# smallptGPU & smallptCPU Makefile
#

CC=nvcc

CCFLAGS=-g -I$(ATISTREAMSDKROOT)/inc 
#-v

LIB = -lm -lGLU -lGL -lglut -lGLEW -lcurand --ptxas-options=-v

smallptCPU: device.cu smallptCPU.c displayfunc.c vec.h camera.h geom.h displayfunc.h simplernd.h scene.h geomfunc.h
	$(CC) $(CCFLAGS) -DSMALLPT_CPU -o smallptCPU device.cu smallptCPU.c displayfunc.c MersenneTwister_kernel.cu $(LIB)
clean:
	rm -rf smallptCPU smallptGPU image.ppm SmallptGPU-v1.6 smallptgpu-v1.6.tgz preprocessed_rendering_kernel.cl

cleanobj:
	rm -f core *.o a.out
.f.o:
	$(FC) $(FFLAGS) -c $*.f
.c.o:
	$(CC) $(INC) $(CFLAGS) -c $*.c

