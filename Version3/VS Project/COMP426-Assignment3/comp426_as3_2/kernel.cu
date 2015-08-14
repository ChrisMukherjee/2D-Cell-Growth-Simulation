
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <string>

// OpenGL Graphics includes
#include "GL/glew.h"
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include "GL/freeglut.h" 
#endif

// Define states for cells
#define HEALTHY  0
#define CANCER   1
#define MEDICINE 2

// 2D area of 1024 x 768 cells
const int g_windowWidth = 1024;
const int g_windowHeight = 768;
int g_quad_read[g_windowWidth][g_windowHeight];
int g_quad_write[g_windowWidth][g_windowHeight];

// Update every 1/30th second
const int g_updateTime = 1.0 / 30.0 * 1000.0;

// At least 25% of cells initialized as cancer cells
const int g_initialCancer = g_windowWidth * g_windowHeight * 0.26;

const int g_font = (int)GLUT_BITMAP_TIMES_ROMAN_24;

cudaError_t updateWithCuda();

__global__ void updateKernel(int *devRead, int *devWrite)
{
	/**
	@Desc : Updates each cell state
	@param1 : pointer to read array
	@param2 : pointer to write array
	*/

	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (devRead[x*g_windowHeight + y] == HEALTHY || devRead[x*g_windowHeight + y] == CANCER) {
		int _numSurrounded = 0;
		int _before = 0;
		int _after = 0;

		// If a healthy cell is surrounded by >= 6 cancer cells,
		// it becomes a cancer cell
		if (devRead[x*g_windowHeight + y] == HEALTHY) {
			_before = CANCER;
			_after = CANCER;
		}
		// If a cancer cell is surrounded by >= 6 medicine cells,
		// it becomes a healthy cell
		else if (devRead[x*g_windowHeight + y] == CANCER) {
			_before = MEDICINE;
			_after = HEALTHY;
		}

		// Check the states of the surrounding cells
		if (x > 0 && y > 0) {
			if (devRead[(x - 1)*g_windowHeight + (y - 1)] == _before)
				_numSurrounded++;
		}
		if (y > 0) {
			if (devRead[x*g_windowHeight + (y - 1)] == _before)
				_numSurrounded++;
		}
		if (x < (g_windowWidth - 1) && y > 0) {
			if (devRead[(x + 1)*g_windowHeight + (y - 1)] == _before)
				_numSurrounded++;
		}
		if (x > 0) {
			if (devRead[(x - 1)*g_windowHeight + y] == _before)
				_numSurrounded++;
		}
		if (x < (g_windowWidth - 1)) {
			if (devRead[(x + 1)*g_windowHeight + y] == _before)
				_numSurrounded++;
		}
		if (x > 0 && y < (g_windowHeight - 1)) {
			if (devRead[(x - 1)*g_windowHeight + (y + 1)] == _before)
				_numSurrounded++;
		}
		if (y < (g_windowHeight - 1)) {
			if (devRead[x*g_windowHeight + (y + 1)] == _before)
				_numSurrounded++;
		}
		if (x < (g_windowWidth - 1) && y < (g_windowHeight - 1)) {
			if (devRead[(x + 1)*g_windowHeight + (y + 1)] == _before)
				_numSurrounded++;
		}
		// Change state if surrounded by >= 6 of a certain cell
		if (_numSurrounded >= 6) {
			devWrite[x*g_windowHeight + y] = _after;
		}
	}
}

cudaError_t updateWithCuda()
{
	/**
	@Desc : Helper function for using CUDA to update cells in parallel. Launches the CUDA kernel
	*/

	int *dev_read = 0;
    int *dev_write = 0;
	std::size_t *pitch_read = new std::size_t;
	std::size_t *pitch_write = new std::size_t;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for arrays
    cudaStatus = cudaMallocPitch(&dev_read, pitch_read, g_windowWidth * sizeof(std::size_t), g_windowHeight * sizeof(std::size_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
    cudaStatus = cudaMallocPitch(&dev_write, pitch_write, g_windowWidth * sizeof(std::size_t), g_windowHeight * sizeof(std::size_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy arrays from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_read, g_quad_read, (g_windowWidth*g_windowHeight) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    cudaStatus = cudaMemcpy(dev_write, g_quad_write, (g_windowWidth*g_windowHeight) * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	dim3 dimBlock(16, 32);
	dim3 dimGrid;
	dimGrid.x = (1024 + dimBlock.x - 1) / dimBlock.x;
	dimGrid.y = (768 + dimBlock.y - 1) / dimBlock.y;

    // Launch a kernel on the GPU with one thread for each element.
    updateKernel<<<dimGrid, dimBlock>>>(dev_read, dev_write);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// Copy array from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(g_quad_write, dev_write, (g_windowWidth*g_windowHeight) * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_read);
    cudaFree(dev_write);
    
    return cudaStatus;
}

void Update(int value)
{
	/**
	@Desc : Function that uses CUDA to update the cells in parallal, and then calls itself (to update again)
	@param1 : unused parameter that is passed by the glutTimerFunc
	*/

	// Update read array with current data from write array before each new update
	for (int i = 0; i < g_windowWidth; ++i) {
		for (int j = 0; j < g_windowHeight; ++j) {
			g_quad_read[i][j] = g_quad_write[i][j];
		}
	}

	// Update cells in parallel
    cudaError_t cudaStatus = updateWithCuda();

	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "updateWithCuda failed!");
        return;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return;
    }

	glutPostRedisplay();
	glutTimerFunc(g_updateTime, Update, 0);
}

void RenderBitmapString(float x, float y, void *font, const char *string)
{
	/**
	@Desc : Renders bitmap strings to display text on screen
	@param1 : x position of where text should be displayed
	@param2 : y position of where text should be displayed
	@param3 : font to be used
	@param4 : string text to be displayed on screen
	*/

	const char *c;
	glRasterPos2f(x, y);
	for (c = string; *c != '\0'; c++) {
		glutBitmapCharacter(font, *c);
	}
}

void Display()
{
	/**
	@Desc : Displays the cells and text in a window on screen
	*/

	// Display the cells using OpenGL
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, g_windowWidth, g_windowHeight, 0);

	glClearColor(1, 1, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_QUADS);
	int _healthyCount = 0;
	int _cancerCount = 0;
	int _medicineCount = 0;
	for (int x = 0; x < g_windowWidth; x++)
	{
		for (int y = 0; y < g_windowHeight; y++)
		{
			if (g_quad_read[x][y] == HEALTHY)
			{
				// Healthy cells are green
				glColor3f(0, 0.5, 0);
				_healthyCount++;
			}
			else if (g_quad_read[x][y] == CANCER)
			{
				// Cancer cells are red
				glColor3f(1, 0, 0);
				_cancerCount++;
			}
			else if (g_quad_read[x][y] == MEDICINE)
			{
				// Medicine cells are yellow
				glColor3f(1, 1, 0);
				_medicineCount++;
			}
			glVertex2f(x, y);
			glVertex2f(x + 1, y);
			glVertex2f(x + 1, y + 1);
			glVertex2f(x, y + 1);
		}
	}
	glEnd();

	std::string _hCount = std::to_string(static_cast<long long>(_healthyCount));
	const char * _hc = _hCount.c_str();
	std::string _cCount = std::to_string(static_cast<long long>(_cancerCount));
	const char * _cc = _cCount.c_str();
	std::string _mCount = std::to_string(static_cast<long long>(_medicineCount));
	const char * _mc = _mCount.c_str();

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glColor3f(0, 0, 0);
	// Display the number of each type of cell
	RenderBitmapString(0, 30, (void *)g_font, "Healthy: ");
	RenderBitmapString(0, 50, (void *)g_font, _hc);
	RenderBitmapString(0, 100, (void*)g_font, "Cancer: ");
	RenderBitmapString(0, 120, (void *)g_font, _cc);
	RenderBitmapString(0, 170, (void *)g_font, "Medicine: ");
	RenderBitmapString(0, 190, (void *)g_font, _mc);
	glPopMatrix();

	glutSwapBuffers();
}

void Initialize()
{
	/**
	@Desc : Initialization function for glut
	*/

	glMatrixMode(GL_PROJECTION);
	glViewport(0, 0, g_windowWidth, g_windowHeight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	GLfloat aspect = (GLfloat)g_windowWidth / g_windowHeight;
	gluPerspective(45, aspect, 0.1f, 10.0f);
	glClearColor(0.0, 0.0, 0.0, 0.0);
}

void MouseClicks(int button, int state, int x, int y)
{
	/**
	@Desc : Function that handles mouse buttons being clicked
	@param1 : mouse button that was clicked
	@param2 : state of button that was clicked
	@param3 : x position of pointer when mouse was clicked
	@param4 : y position of pointer when mouse was clicked
	*/

	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		// If medicine is injected on a cancer cell,
		// the medicine is absorbed and the cell turns into a healthy cell
		if (g_quad_read[x][y] == CANCER) {
			g_quad_write[x][y] = HEALTHY;
		}
		// If medicine is injected on a healthy or medicine cell,
		// the medicine is not absorbed and propagates radially outwards by one cell
		else {
			g_quad_write[x][y] = MEDICINE;
			if (x > 0 && y > 0)
				g_quad_write[x - 1][y - 1] = MEDICINE;
			if (y > 0)
				g_quad_write[x][y - 1] = MEDICINE;
			if (x < (g_windowWidth - 1) && y > 0)
				g_quad_write[x + 1][y - 1] = MEDICINE;
			if (x > 0)
				g_quad_write[x - 1][y] = MEDICINE;
			if (x < (g_windowWidth - 1))
				g_quad_write[x + 1][y] = MEDICINE;
			if (x > 0 && y < (g_windowHeight - 1))
				g_quad_write[x - 1][y + 1] = MEDICINE;
			if (y < (g_windowHeight - 1))
				g_quad_write[x][y + 1] = MEDICINE;
			if (x < (g_windowWidth - 1) && y < (g_windowHeight - 1))
				g_quad_write[x + 1][y + 1] = MEDICINE;
		}
	}
}

void Keyboard(unsigned char key, int mousePositionX, int mousePositionY)
{
	/**
	@Desc : Function that handles keyboard buttons being pressed
	@param1 : key that was pressed
	@param2 : x position of mouse pointer
	@param3 : y position of mouse pointer
	*/

	switch (key)
	{
	// Escape key
	case 27:
		exit ( 0 );
		break;

	default:
		break;
	}
}

int main(int argc, char **argv)
{
	/**
	@Desc : Main control thread
	*/

	// initialize
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowSize(g_windowWidth, g_windowHeight);
	glutCreateWindow("2D Cell Growth Simulation");

	// Initialize all cells as healthy cells
	for (int i = 0; i < 1024; i++)
	{
		for (int j = 0; j < 768; j++)
		{
			g_quad_write[i][j] = HEALTHY;
		}
	}

	// Initialize random seed
	srand(time(NULL));

	// Change at least 25% of cells to cancer cells
	for (int i = 0; i <= g_initialCancer; i++)
	{
		int x = rand() % 1024;
		int y = rand() % 768;
		if (g_quad_write[x][y] == CANCER)
			i--;
		else
			g_quad_write[x][y] = CANCER;
	}

	glutDisplayFunc(Display);
	glutIdleFunc(Display);
	glutMouseFunc(MouseClicks);
	glutKeyboardFunc(Keyboard);
	glutTimerFunc(g_updateTime, Update, 0);
	Initialize();

	glutMainLoop();
	return 0;
}