
//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

//########################################################################
//### Hotspot
//########################################################################

// Parameters
#define GRID_ROWS 512                                       // Number of rows in the grid (positive integer)
#define GRID_COLS 512                                       // Number of columns in the grid (positive integer)
#define SIM_TIME 10                                         // Number of iterations
#define TEMPERATURE_FILE "data/hotspot_temperature_512.txt" // Name of the file containing the initial temperature values of each cell
#define POWER_FILE "data/hotspot_power_512.txt"             // Name of the file containing the dissipated power values of each cell

// Defines
#define STRING_SIZE 256          // Length of the strings in the temperature and power files
#define MAX_PD (3.0e6)           // Maximum power density possible (say 300W for a 10mm x 10mm chip)
#define PRECISION 0.001          // Required precision in degrees
#define SPEC_HEAT_SI 1.75e6      // 
#define K_SI 100                 // 
#define FACTOR_CHIP 0.5          // Capacitance fitting factor
#define T_CHIP 0.0005            // Chip temperature
#define CHIP_HEIGHT 0.016        // Chip height
#define CHIP_WIDTH 0.016         // Chip width
#define AMB_TEMP 80.0            // Ambient temperature, assuming no package at all

//########################################################################
//### Srad
//########################################################################

// Defines
#define ROWS 256    // Number of ROWS in the domain
#define COLS 256    // Number of COLS in the domain
#define R1 0        // y1 position of the speckle
#define R2 31       // y2 position of the speckle
#define C1 0        // x1 position of the speckle
#define C2 31       // x2 position of the speckle
#define LAMBDA 0.5  // Lambda value
#define NITER 4     // Number of iterations

//########################################################################
//### Pathfinder
//########################################################################

// Configuration
#define PATHCOLS 100000
#define PATHROWS 100

// Seed
#define M_SEED 9

//########################################################################
//### Kmeans
//########################################################################

#define SIZE (494020)
#define NUM_CLUSTERS 20
#define DIMENSIONS 2
#define THRESHOLD 0.001

//########################################################################
//### BFS
//########################################################################

// Settings
#define MAX_NODES 1000000

// Config
#define FILENAME "/home/cnugteren/software/rodinia_2.4/data/bfs/graph1MW_6.txt"

//########################################################################