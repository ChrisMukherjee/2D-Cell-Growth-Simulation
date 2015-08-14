#include <windows.h>
#include <time.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include "tbb/task_scheduler_init.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include <string>

// Define states for cells
#define HEALTHY  0
#define CANCER   1
#define MEDICINE 2

// 2D area of 1024 x 768 cells
const int g_windowWidth = 1024;
const int g_windowHeight = 768;
int g_quad[g_windowWidth][g_windowHeight];

// Update every 1/30th second
const int g_updateTime = 1.0 / 30.0 * 1000.0;

// At least 25% of cells initialized as cancer cells
const int g_initialCancer = g_windowWidth * g_windowHeight * 0.26;

const int g_font = (int)GLUT_BITMAP_TIMES_ROMAN_24;

const size_t init_size = 0;

void HealSurroundingMedicine(int x, int y)
{
	/**
	@Desc : Heals all surrounding medicine cells when a cancer cell turns into a healthy cell
	@param1 : x position of current cell
	@param2 : y position of current cell
	*/

	g_quad[x][y] = HEALTHY;
	if (x > 0 && y > 0) {
		if (g_quad[x - 1][y - 1] == MEDICINE)
			HealSurroundingMedicine(x - 1, y - 1);
	}
	if (y > 0) {
		if (g_quad[x][y - 1] == MEDICINE)
			HealSurroundingMedicine(x, y - 1);
	}
	if (x < (g_windowWidth - 1) && y > 0) {
		if (g_quad[x + 1][y - 1] == MEDICINE)
			HealSurroundingMedicine(x + 1, y - 1);
	}
	if (x > 0) {
		if (g_quad[x - 1][y] == MEDICINE)
			HealSurroundingMedicine(x - 1, y);
	}
	if (x < (g_windowWidth - 1)) {
		if (g_quad[x + 1][y] == MEDICINE)
			HealSurroundingMedicine(x + 1, y);
	}
	if (x > 0 && y < (g_windowHeight - 1)) {
		if (g_quad[x - 1][y + 1] == MEDICINE)
			HealSurroundingMedicine(x - 1, y + 1);
	}
	if (y < (g_windowHeight - 1)) {
		if (g_quad[x][y + 1] == MEDICINE)
			HealSurroundingMedicine(x, y + 1);
	}
	if (x < (g_windowWidth - 1) && y < (g_windowHeight - 1)) {
		if (g_quad[x + 1][y + 1] == MEDICINE)
			HealSurroundingMedicine(x + 1, y + 1);
	}
}

void UpdateState(int x, int y, int state)
{
	/**
	@Desc : Updates each cell state (called by each computational thread, which are generated using TBB)
	@param1 : x position of current cell
	@param2 : y position of current cell
	@param3 : state of current cell
	*/

	if (state == HEALTHY || state == CANCER) {
		int _numSurrounded = 0;
		int _before = 0;
		int _after = 0;

		// If a healthy cell is surrounded by >= 6 cancer cells,
		// it becomes a cancer cell
		if (state == HEALTHY) {
			_before = CANCER;
			_after = CANCER;
		}
		// If a cancer cell is surrounded by >= 6 medicine cells,
		// it becomes a healthy cell
		else if (state == CANCER) {
			_before = MEDICINE;
			_after = HEALTHY;
		}

		// Check the states of the surrounding cells
		if (x > 0 && y > 0) {
			if (g_quad[x - 1][y - 1] == _before)
				_numSurrounded++;
		}
		if (y > 0) {
			if (g_quad[x][y - 1] == _before)
				_numSurrounded++;
		}
		if (x < (g_windowWidth - 1) && y > 0) {
			if (g_quad[x + 1][y - 1] == _before)
				_numSurrounded++;
		}
		if (x > 0) {
			if (g_quad[x - 1][y] == _before)
				_numSurrounded++;
		}
		if (x < (g_windowWidth - 1)) {
			if (g_quad[x + 1][y] == _before)
				_numSurrounded++;
		}
		if (x > 0 && y < (g_windowHeight - 1)) {
			if (g_quad[x - 1][y + 1] == _before)
				_numSurrounded++;
		}
		if (y < (g_windowHeight - 1)) {
			if (g_quad[x][y + 1] == _before)
				_numSurrounded++;
		}
		if (x < (g_windowWidth - 1) && y < (g_windowHeight - 1)) {
			if (g_quad[x + 1][y + 1] == _before)
				_numSurrounded++;
		}
		// Change state if surrounded by >= 6 of a certain cell
		if (_numSurrounded >= 6) {
			// When a cancer cell becomes a healthy cell,
			// all the surrounding medicine cells also become healthy cells
			if (state == CANCER)
				HealSurroundingMedicine(x, y);
			else
				g_quad[x][y] = _after;
		}
	}
}

class DoUpdate
{
	/**
	@Desc : Class with overloaded parenthesis () operator
	*/
	int *startX, *endX, *startY, *endY;
public:
	DoUpdate(int *sX, int *eX, int *sY, int *eY)
		: startX(sX), endX(eX), startY(sY), endY(eY) { }

	// overload () so it starts updating the cell states
	void operator()(const tbb::blocked_range2d<size_t>& r) const
	{
		/**
		@Desc : Overloaded parenthesis () operator
		@param1 : TBB 2D blocked range
		*/

		// Update each cell that the current thread manages
		for (size_t i = r.cols().begin(); i != r.cols().end(); ++i)
		{
			for (size_t j = r.rows().begin(); j != r.rows().end(); ++j)
				UpdateState(i, j, g_quad[i][j]);
		}
	}
};

void Update(int value)
{
	/**
	@Desc : Function that uses TBB to initialize a task scheduler and perform a parallel loop, and then calls itself (to update again)
	@param1 : unused parameter that is passed by the glutTimerFunc
	*/

	// allocate storage for vectors
	int *startX, *endX, *startY, *endY;
	startX = (int*)malloc(init_size*sizeof(int));
	endX = (int*)malloc(init_size*sizeof(int));
	startY = (int*)malloc(init_size*sizeof(int));
	endY = (int*)malloc(init_size*sizeof(int));

	tbb::task_scheduler_init init;

	*startX = 0, *endX = g_windowWidth, *startY = 0, *endY = g_windowHeight;
	tbb::parallel_for(tbb::blocked_range2d<size_t>(*startX, *endX, 1, *startY, *endY, 1), DoUpdate(startX, endX, startY, endY), tbb::auto_partitioner());

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
			if (g_quad[x][y] == HEALTHY)
			{
				// Healthy cells are green
				glColor3f(0, 0.5, 0);
				_healthyCount++;
			}
			else if (g_quad[x][y] == CANCER)
			{
				// Cancer cells are red
				glColor3f(1, 0, 0);
				_cancerCount++;
			}
			else if (g_quad[x][y] == MEDICINE)
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

	std::string _hCount = std::to_string(_healthyCount);
	const char * _hc = _hCount.c_str();
	std::string _cCount = std::to_string(_cancerCount);
	const char * _cc = _cCount.c_str();
	std::string _mCount = std::to_string(_medicineCount);
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
		if (g_quad[x][y] == CANCER) {
			g_quad[x][y] = HEALTHY;
		}
		// If medicine is injected on a healthy or medicine cell,
		// the medicine is not absorbed and propagates radially outwards by one cell
		else {
			g_quad[x][y] = MEDICINE;
			if (x > 0 && y > 0)
				g_quad[x - 1][y - 1] = MEDICINE;
			if (y > 0)
				g_quad[x][y - 1] = MEDICINE;
			if (x < (g_windowWidth - 1) && y > 0)
				g_quad[x + 1][y - 1] = MEDICINE;
			if (x > 0)
				g_quad[x - 1][y] = MEDICINE;
			if (x < (g_windowWidth - 1))
				g_quad[x + 1][y] = MEDICINE;
			if (x > 0 && y < (g_windowHeight - 1))
				g_quad[x - 1][y + 1] = MEDICINE;
			if (y < (g_windowHeight - 1))
				g_quad[x][y + 1] = MEDICINE;
			if (x < (g_windowWidth - 1) && y < (g_windowHeight - 1))
				g_quad[x + 1][y + 1] = MEDICINE;
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
			g_quad[i][j] = HEALTHY;
		}
	}

	// Initialize random seed
	srand(time(NULL));

	// Change at least 25% of cells to cancer cells
	for (int i = 0; i <= g_initialCancer; i++)
	{
		int x = rand() % 1024;
		int y = rand() % 768;
		if (g_quad[x][y] == CANCER)
			i--;
		else
			g_quad[x][y] = CANCER;
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