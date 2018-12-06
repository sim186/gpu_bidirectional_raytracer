#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef WIN32
#define _USE_MATH_DEFINES
#endif
//#include <math.h>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/time.h>
#elif defined (WIN32)
#include <windows.h>
#else
        Unsupported Platform !!!
#endif

#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "camera.h"
#include "geom.h"
#include "displayfunc.h"

extern void ReInit(const int);
extern void ReInitScene();
extern void UpdateRendering2();
extern void UpdateRendering();
extern void UpdateCamera();
extern void savePPM(int numbe);


extern Camera camera;
extern Sphere *spheres;
extern Vec *colors;
extern Vec *colors2;
extern unsigned int sphereCount;
extern GLuint pbo;
extern GLuint textureID;
extern uchar4 *pixels_buf;

int amiSmallptCPU;
extern int flag=1;
extern int allFlag;
float animTime=0.0f;
float animInc=0.1f;
int animFlag=1;

int width = 640;
float invWidth;
int height = 480;
float invHeight;
//unsigned int *pixels;
uchar4 *pixels;
unsigned int *counter;
char captionBuffer[256];

static int printHelp = 1;
static int currentSphere;

double WallClockTime() {
#if defined(__linux__) || defined(__APPLE__)
	struct timeval t;
	gettimeofday(&t, NULL);

	return t.tv_sec + t.tv_usec / 1000000.0;
#elif defined (WIN32)
	return GetTickCount() / 1000.0;
#else
	Unsupported Platform !!!
#endif
}

static void PrintString(void *font, const char *string) {
	int len, i;

	len = (int)strlen(string);
	for (i = 0; i < len; i++)
		glutBitmapCharacter(font, string[i]);
}

static void PrintHelp() {
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(0.f, 0.f, 0.5f, 0.5f);
	glRecti(40, 40, 600, 440);

	glColor3f(1.f, 1.f, 1.f);
	glRasterPos2i(300, 420);
	PrintString(GLUT_BITMAP_HELVETICA_18, "Help");

	glRasterPos2i(60, 390);
	PrintString(GLUT_BITMAP_HELVETICA_18, "h - toggle Help");
	glRasterPos2i(60, 360);
	PrintString(GLUT_BITMAP_HELVETICA_18, "arrow Keys - rotate camera left/right/up/down");
	glRasterPos2i(60, 330);
	PrintString(GLUT_BITMAP_HELVETICA_18, "a and d - move camera left and right");
	glRasterPos2i(60, 300);
	PrintString(GLUT_BITMAP_HELVETICA_18, "w and s - move camera forward and backward");
	glRasterPos2i(60, 270);
	PrintString(GLUT_BITMAP_HELVETICA_18, "r and f - move camera up and down");
	glRasterPos2i(60, 240);
	PrintString(GLUT_BITMAP_HELVETICA_18, "PageUp and PageDown - move camera target up and down");
	glRasterPos2i(60, 210);
	PrintString(GLUT_BITMAP_HELVETICA_18, "+ and - - to select next/previous object");
	glRasterPos2i(60, 180);
	PrintString(GLUT_BITMAP_HELVETICA_18, "2, 3, 4, 5, 6, 8, 9 - to move selected object");

	glDisable(GL_BLEND);
}

void ReadScene(char *fileName) {
	fprintf(stderr, "Reading scene: %s\n", fileName);

	FILE *f = fopen(fileName, "r");
	if (!f) {
		fprintf(stderr, "Failed to open file: %s\n", fileName);
		exit(-1);
	}

	/* Read the camera position */
	int c = fscanf(f,"camera %f %f %f  %f %f %f\n",
			&camera.orig.x, &camera.orig.y, &camera.orig.z,
			&camera.target.x, &camera.target.y, &camera.target.z);
	if (c != 6) {
		fprintf(stderr, "Failed to read 6 camera parameters: %d\n", c);
		exit(-1);
	}

	/* Read the sphere count */
	c = fscanf(f,"size %u\n", &sphereCount);
	if (c != 1) {
		fprintf(stderr, "Failed to read sphere count: %d\n", c);
		exit(-1);
	}
	fprintf(stderr, "Scene size: %d\n", sphereCount);

	/* Read all spheres */
	spheres = (Sphere *)malloc(sizeof(Sphere) * sphereCount);
	unsigned int i;
	for (i = 0; i < sphereCount; i++) {
		Sphere *s = &spheres[i];
		int mat;
		int c = fscanf(f,"sphere %f  %f %f %f  %f %f %f  %f %f %f  %d\n",
				&s->rad,
				&s->p.x, &s->p.y, &s->p.z,
				&s->e.x, &s->e.y, &s->e.z,
				&s->c.x, &s->c.y, &s->c.z,
				&mat);
		/*if(!viszero(s->e)){
			vnorm(s->e);
			vsmul(s->e,1.6,s->e);
		}*/
		switch (mat) {
			case 0:
				s->refl = DIFF;
				break;
			case 1:
				s->refl = SPEC;
				break;
			case 2:
				s->refl = REFR;
				break;
			case 3:
				s->refl = LITE;
				break;
			default:
				fprintf(stderr, "Failed to read material type for sphere #%d: %d\n", i, mat);
				exit(-1);
				break;
		}
		if (c != 11) {
			fprintf(stderr, "Failed to read sphere #%d: %d\n", i, c);
			exit(-1);
		}
	}

	fclose(f);
}

void UpdateCamera() {
	vsub(camera.dir, camera.target, camera.orig);
	vnorm(camera.dir);

	const Vec up = {0.f, 1.f, 0.f};
	const float fov = (M_PI / 180.f) * 45.f;
	vxcross(camera.x, camera.dir, up);
	vnorm(camera.x);
	vsmul(camera.x, width * fov / height, camera.x);

	vxcross(camera.y, camera.x, camera.dir);
	vnorm(camera.y);
	vsmul(camera.y, fov, camera.y);
}

void idleFunc(void) {
	if (allFlag){
		sleep(20);
		exit(0);
	}

    //printf("IDLEFUNC\n");
    printf("ff=%d\n",flag);
	cudaError_t error=cudaGLMapBufferObject((void**)&pixels_buf, pbo);
	if (error != cudaSuccess) {
				fprintf(stderr, "Map Buffer failed: %s\n",
				cudaGetErrorString(error));
	}
	if (flag==1){
        printf("UP2\n");
		UpdateRendering2();
	}
	//printf("IDLEFUNC-dopo flag\n");
	if (flag>1){
		//printf("UP\n");
		//flag=1;
		UpdateRendering();
		//UpdateRendering2();
	}
	error=cudaGLUnmapBufferObject(pbo);
	if (error != cudaSuccess) {
				fprintf(stderr, "Unmap Buffer failed: %s\n",
				cudaGetErrorString(error));
	}
 //if(currentSample==100) {savePPM(1); for(;;) printf("Loop\n");}
	glutPostRedisplay();
}

/*void GouraudPixel(int *writePixels) {
	int i,j,k1,k2;
	Vec temp;
	for (i=0; i<height-1; i++){
		for (j=0; j<width-1; j++){
			k1=i*(width)+j;
			k2=i*(width-1)+j;
			temp.x=(colors[k1].x+colors2[k1].x+colors[k1+1].x+colors2[k1+1].x+colors[k1+1+width].x+colors2[k1+1+width].x+colors[k1+width].x+colors2[k1+width].x)/4;
			temp.y=(colors[k1].y+colors2[k1].y+colors[k1+1].y+colors2[k1+1].y+colors[k1+1+width].y+colors2[k1+1+width].y+colors[k1+width].y+colors2[k1+width].y)/4;
			temp.z=(colors[k1].z+colors2[k1].z+colors[k1+1].z+colors2[k1+1].z+colors[k1+1+width].z+colors2[k1+1+width].z+colors[k1+width].z+colors2[k1+width].z)/4;
		//	writePixels[i*(width-1)+j]=(pixels[i*(width-1)+j]+pixels[i*(width-1)+j+1]+pixels[(i+1)*(width-1)+j]+pixels[(i+1)*(width-1)+j+1])/4;
			writePixels[k2]=toInt(temp.x) |
						(toInt(temp.y) << 8) |
						(toInt(temp.z) << 16);
		}
	}
}*/
			
void displayFunc(void) {
	glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pbo);
	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0, 0);
	//glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
	//glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, &pbo);

	glBindTexture(GL_TEXTURE_2D, textureID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, 
		  GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBegin(GL_QUADS);
  	glTexCoord2f(0.0f,0.0f); glVertex3f(0.0f,0.0f,0.0f);
  	glTexCoord2f(0.0f,1.0f); glVertex3f(0.0f,1.0f,0.0f);
  	glTexCoord2f(1.0f,1.0f); glVertex3f(1.0f,1.0f,0.0f);
  	glTexCoord2f(1.0f,0.0f); glVertex3f(1.0f,0.0f,0.0f);
  	glEnd();

	// Title
	/*glColor3f(1.f, 1.f, 1.f);
	glRasterPos2i(4, height - 16);
	if (amiSmallptCPU)
		PrintString(GLUT_BITMAP_HELVETICA_18, "SmallptCPU v1.6 (Written by David Bucciarelli)");
	else
		PrintString(GLUT_BITMAP_HELVETICA_18, "SmallptGPU v1.6 (Written by David Bucciarelli)");*/

	// Caption line 0
	glColor3f(1.f, 1.f, 1.f);
	glRasterPos2i(4, 10);
	PrintString(GLUT_BITMAP_HELVETICA_18, captionBuffer);

	if (printHelp) {
		glPushMatrix();
		glLoadIdentity();
		glOrtho(-0.5, 639.5, -0.5, 479.5, -1.0, 1.0);

		PrintHelp();

		glPopMatrix();
	}

	glFlush();
	//glFinish();
	glutSwapBuffers();
	if(animFlag) {
    		glutPostRedisplay();
    		animTime += animInc;
  	}

}

void reshapeFunc(int newWidth, int newHeight) {
	width = newWidth;
	height = newHeight;

	/*glViewport(0, 0, width, height);
	glLoadIdentity();
	glOrtho(0.f, width - 1.f, 0.f, height - 1.f, -1.f, 1.f);*/

	glViewport(0, 0, width, height);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);


	//ReInit(1);

	glutPostRedisplay();
}



#define MOVE_STEP 10.0f
#define ROTATE_STEP (2.f * M_PI / 180.f)
void keyFunc(unsigned char key, int x, int y) {
	switch (key) {
		case 'p': {
			savePPM(0);
			break;
		}
		case 27: /* Escape key */
			fprintf(stderr, "Done.\n");
			exit(0);
			break;
		case ' ': /* Refresh display */
			ReInit(1);
			break;
		case 'a': {
			Vec dir = camera.x;
			vnorm(dir);
			vsmul(dir, -MOVE_STEP, dir);
			vadd(camera.orig, camera.orig, dir);
			vadd(camera.target, camera.target, dir);
			ReInit(1);
			break;
		}
		case 'd': {
			Vec dir = camera.x;
			vnorm(dir);
			vsmul(dir, MOVE_STEP, dir);
			vadd(camera.orig, camera.orig, dir);
			vadd(camera.target, camera.target, dir);
			ReInit(1);
			break;
		}
		case 'w': {
			Vec dir = camera.dir;
			vsmul(dir, MOVE_STEP, dir);
			vadd(camera.orig, camera.orig, dir);
			vadd(camera.target, camera.target, dir);
			ReInit(1);
			break;
		}
		case 's': {
			Vec dir = camera.dir;
			vsmul(dir, -MOVE_STEP, dir);
			vadd(camera.orig, camera.orig, dir);
			vadd(camera.target, camera.target, dir);
			ReInit(1);
			break;
		}
		case 'r':
			camera.orig.y += MOVE_STEP;
			camera.target.y += MOVE_STEP;
			ReInit(1);
			break;
		case 'f':
			camera.orig.y -= MOVE_STEP;
			camera.target.y -= MOVE_STEP;
			ReInit(1);
			break;
		case '+':
			currentSphere = (currentSphere + 1) % sphereCount;
			fprintf(stderr, "Selected sphere %d (%f %f %f)\n", currentSphere,
					spheres[currentSphere].p.x, spheres[currentSphere].p.y, spheres[currentSphere].p.z);
			ReInitScene();
			break;
		case '-':
			currentSphere = (currentSphere + (sphereCount - 1)) % sphereCount;
			fprintf(stderr, "Selected sphere %d (%f %f %f)\n", currentSphere,
					spheres[currentSphere].p.x, spheres[currentSphere].p.y, spheres[currentSphere].p.z);
			ReInitScene();
			break;
		case '4':
			spheres[currentSphere].p.x -= 0.5f * MOVE_STEP;
			ReInitScene();
			break;
		case '6':
			spheres[currentSphere].p.x += 0.5f * MOVE_STEP;
			ReInitScene();
			break;
		case '8':
			spheres[currentSphere].p.z -= 0.5f * MOVE_STEP;
			ReInitScene();
			break;
		case '2':
			spheres[currentSphere].p.z += 0.5f * MOVE_STEP;
			ReInitScene();
			break;
		case '9':
			spheres[currentSphere].p.y += 0.5f * MOVE_STEP;
			ReInitScene();
			break;
		case '3':
			spheres[currentSphere].p.y -= 0.5f * MOVE_STEP;
			ReInitScene();
			break;
		case 'h':
			printHelp = (!printHelp);
			break;
	    case 'i':
	        printf("Origine:(%.1f,%1.f,%1.f)\n\nTarget:(%.1f,%1.f,%1.f)",camera.orig.x,camera.orig.y,camera.orig.z,camera.target.x,camera.target.y,camera.target.z);
	        break;
		default:
			break;
	}
}

void specialFunc(int key, int x, int y) {
	switch (key) {
		case GLUT_KEY_UP: {
			Vec t = camera.target;
			vsub(t, t, camera.orig);
			t.y = t.y * cos(-ROTATE_STEP) + t.z * sin(-ROTATE_STEP);
			t.z = -t.y * sin(-ROTATE_STEP) + t.z * cos(-ROTATE_STEP);
			vadd(t, t, camera.orig);
			camera.target = t;
			ReInit(1);
			break;
		}
		case GLUT_KEY_DOWN: {
			Vec t = camera.target;
			vsub(t, t, camera.orig);
			t.y = t.y * cos(ROTATE_STEP) + t.z * sin(ROTATE_STEP);
			t.z = -t.y * sin(ROTATE_STEP) + t.z * cos(ROTATE_STEP);
			vadd(t, t, camera.orig);
			camera.target = t;
			ReInit(1);
			break;
		}
		case GLUT_KEY_LEFT: {
			Vec t = camera.target;
			vsub(t, t, camera.orig);
			t.x = t.x * cos(-ROTATE_STEP) - t.z * sin(-ROTATE_STEP);
			t.z = t.x * sin(-ROTATE_STEP) + t.z * cos(-ROTATE_STEP);
			vadd(t, t, camera.orig);
			camera.target = t;
			ReInit(1);
			break;
		}
		case GLUT_KEY_RIGHT: {
			Vec t = camera.target;
			vsub(t, t, camera.orig);
			t.x = t.x * cos(ROTATE_STEP) - t.z * sin(ROTATE_STEP);
			t.z = t.x * sin(ROTATE_STEP) + t.z * cos(ROTATE_STEP);
			vadd(t, t, camera.orig);
			camera.target = t;
			ReInit(1);
			break;
		}
		case GLUT_KEY_PAGE_UP:
			camera.target.y += MOVE_STEP;
			ReInit(1);
			break;
		case GLUT_KEY_PAGE_DOWN:
			camera.target.y -= MOVE_STEP;
			ReInit(1);
			break;
		default:
			break;
	}
}

void InitGlut(int argc, char *argv[], char *windowTittle) {
	glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    //glutInitDisplayMode(GLUT_RGBA);
    glutInitWindowSize(width, height);

	glutCreateWindow(windowTittle);

    glutReshapeFunc(reshapeFunc);
    glutKeyboardFunc(keyFunc);
    glutSpecialFunc(specialFunc);
    glutDisplayFunc(displayFunc);
	glutIdleFunc(idleFunc);

	glewInit();
	if (! glewIsSupported( "GL_VERSION_2_0 " ) ) {
    		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
  	}

	glViewport(0, 0, width, height);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f);

	//glLoadIdentity();
	//glOrtho(0.f, width - 1.f, 0.f, height - 1.f, -1.f, 1.f);
}
