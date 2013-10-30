//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'Hotspot', taken from
// the Rodinia benchmark suite. For more information on the application or on Bones
// please use the contact information below.
//
// == More information on Hotspot
// Article............http://dx.doi.org/10.1109/TVLSI.2006.876103
// Original code......https://www.cs.virginia.edu/~skadron/wiki/rodinia/
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/hotspot.c
// Authors............Cedric Nugteren
// Last modified on...10-Aug-2012
//

//########################################################################
//### Includes
//########################################################################

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//########################################################################
//### Input parameters
//########################################################################

#define GRID_ROWS 64                                        // Number of rows in the grid (positive integer)
#define GRID_COLS 64                                        // Number of columns in the grid (positive integer)
#define SIM_TIME 2                                          // Number of iterations
#define TEMPERATURE_FILE "data/hotspot_temperature_64.txt"  // Name of the file containing the initial temperature values of each cell
#define POWER_FILE "data/hotspot_power_64.txt"              // Name of the file containing the dissipated power values of each cell

//########################################################################
//### Defines
//########################################################################

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
//### Forward declarations
//########################################################################

void read_input(double* array, const char* filename);

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {
	
	// Declare the loop iterators
	int r,c,iter;
	
	// Declare other/helper variables
	int index;
	double delta;
	int row = GRID_ROWS;
	int col = GRID_COLS;
	double grid_height = CHIP_HEIGHT/row;
	double grid_width = CHIP_WIDTH/col;
	
	// Set domain variables
	double cap = FACTOR_CHIP*SPEC_HEAT_SI*T_CHIP*grid_width*grid_height;
	double Rx = grid_width / (2.0*K_SI*T_CHIP*grid_height);
	double Ry = grid_height / (2.0*K_SI*T_CHIP*grid_width);
	double Rz = T_CHIP / (K_SI*grid_height*grid_width);
	double max_slope = MAX_PD / (FACTOR_CHIP*T_CHIP*SPEC_HEAT_SI);
	double step = PRECISION / max_slope;
	
	// Initialising memory
	printf("\n[hotspot] Initialising memory"); fflush(stdout);
	double* temperature = (double*) calloc(row*col, sizeof(double));
	double* power       = (double*) calloc(row*col, sizeof(double));
	double* result      = (double*) calloc(row*col, sizeof(double));
	
	// Read initial temperature and power arrays
	printf("\n[hotspot] Populating memory"); fflush(stdout);
	read_input(temperature, TEMPERATURE_FILE);
	read_input(power, POWER_FILE);
	
	// Perform the computation a given number of times
	printf("\n[hotspot] Performing the computation %d times",SIM_TIME); fflush(stdout);
	#pragma scop
	for (iter=0; iter<SIM_TIME; iter++) {
		
		// Transient solver driver routine: convert the heat transfer differential equations to difference equations
		// and solve the difference equations by iterating
		for (r=0; r<row; r++) {
			for (c=0; c<col; c++) {
				
				// Corner 1
				if ( (r == 0) && (c == 0) ) {
					delta = (step / cap) * (power[0] + 
						(temperature[1]   - temperature[0]) / Rx + 
						(temperature[col] - temperature[0]) / Ry + 
						(AMB_TEMP         - temperature[0]) / Rz);
				}
				// Corner 2
				else if ((r == 0) && (c == col-1)) {
					delta = (step / cap) * (power[c] + 
						(temperature[c-1]   - temperature[c]) / Rx + 
						(temperature[c+col] - temperature[c]) / Ry + 
						(AMB_TEMP           - temperature[c]) / Rz);
				}
				// Corner 3
				else if ((r == row-1) && (c == col-1)) {
					delta = (step / cap) * (power[r*col+c] + 
						(temperature[r*col+c-1]   - temperature[r*col+c]) / Rx + 
						(temperature[(r-1)*col+c] - temperature[r*col+c]) / Ry + 
						(AMB_TEMP                 - temperature[r*col+c]) / Rz);
				}
				// Corner 4
				else if ((r == row-1) && (c == 0)) {
					delta = (step / cap) * (power[r*col] + 
						(temperature[r*col+1]   - temperature[r*col]) / Rx + 
						(temperature[(r-1)*col] - temperature[r*col]) / Ry + 
						(AMB_TEMP               - temperature[r*col]) / Rz);
				}
				// Edge 1
				else if (r == 0) {
					delta = (step / cap) * (power[c] + 
						(temperature[c+1] + temperature[c-1] - 2.0*temperature[c]) / Rx + 
						(temperature[col+c]                      - temperature[c]) / Ry + 
						(AMB_TEMP                                - temperature[c]) / Rz);
				}
				// Edge 2
				else if (c == col-1) {
					delta = (step / cap) * (power[r*col+c] + 
						(temperature[(r+1)*col+c] + temperature[(r-1)*col+c] - 2.0*temperature[r*col+c]) / Ry + 
						(temperature[r*col+c-1]                                  - temperature[r*col+c]) / Rx + 
						(AMB_TEMP                                                - temperature[r*col+c]) / Rz);
				}
				// Edge 3
				else if (r == row-1) {
					delta = (step / cap) * (power[r*col+c] + 
						(temperature[r*col+c+1] + temperature[r*col+c-1] - 2.0*temperature[r*col+c]) / Rx + 
						(temperature[(r-1)*col+c]                            - temperature[r*col+c]) / Ry + 
						(AMB_TEMP                                            - temperature[r*col+c]) / Rz);
				}
				// Edge 4
				else if (c == 0) {
					delta = (step / cap) * (power[r*col] + 
						(temperature[(r+1)*col] + temperature[(r-1)*col] - 2.0*temperature[r*col]) / Ry + 
						(temperature[r*col+1]                                - temperature[r*col]) / Rx + 
						(AMB_TEMP                                            - temperature[r*col]) / Rz);
				}
				// Inside the chip
				else {
					delta = (step / cap) * (power[r*col+c] + 
						(temperature[(r+1)*col+c] + temperature[(r-1)*col+c] - 2.0*temperature[r*col+c]) / Ry + 
						(temperature[r*col+c+1]   + temperature[r*col+c-1]   - 2.0*temperature[r*col+c]) / Rx + 
						(AMB_TEMP                                                - temperature[r*col+c]) / Rz);
				}
			
			// Update the temperatures
			result[r*col+c] = temperature[r*col+c] + delta;
			}
		}
		
		// Copy the result as the new temperatures
		for (r=0; r<row; r++) {
			for (c=0; c<col; c++) {
				temperature[r*col+c] = result[r*col+c];
			}
		}
	}
	#pragma endscop
	
	// Print the values matrix
	printf("\n[hotspot] Printing the final temperatures:\n\n"); fflush(stdout);
	for (r=0; r<row; r++) {
		for (c=0; c<col; c++) {
			index = r*col+c;
			printf("%6d: %.3lf ", index, temperature[index]);
		}
		printf("\n");
	}
	
	// Clean-up and exit
	printf("\n[hotspot] Completed\n\n"); fflush(stdout);
	free(temperature); free(power); free(result);
	fflush(stdout);
	return 0;
}

//########################################################################
//### Function to read an input file (power or temperature values)
//########################################################################

void read_input(double* array, const char* filename) {
	int r, c;
	char string[STRING_SIZE];
	double value;

	// Open the file
	FILE* file_pointer = fopen(filename, "r");
	if (!file_pointer) { printf("\n[hotspot] Error: file '%s' could not be opened for reading\n\n", filename); fflush(stdout); exit(1); }

	// Process the file
	for (r=0; r<GRID_ROWS; r++) {
		for (c=0; c<GRID_COLS; c++) {
			fgets(string, STRING_SIZE, file_pointer);
			if (feof(file_pointer))                    { printf("\n[hotspot] Error: not enough lines in file '%s'\n\n", filename); fflush(stdout); exit(1); }
			if ((sscanf(string, "%lf", &value) != 1) ) { printf("\n[hotspot] Error: invalid file format for '%s'\n\n", filename); fflush(stdout); exit(1); }
			array[r*GRID_COLS+c] = value;
		}
	}
	
	// Clean-up and return
	fclose(file_pointer);
}

//########################################################################