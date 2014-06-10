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
// Last modified on...06-Jun-2014
//
//########################################################################

// Includes
#include "common.h"

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {
	
	// Declare the loop iterators
	int r,c;
	int iter = 0;
	
	// Declare other/helper variables
	double delta;
	double grid_height = CHIP_HEIGHT/GRID_ROWS;
	double grid_width = CHIP_WIDTH/GRID_COLS;
	
	// Set domain variables
	double cap = FACTOR_CHIP*SPEC_HEAT_SI*T_CHIP*grid_width*grid_height;
	double Rx = grid_width / (2.0*K_SI*T_CHIP*grid_height);
	double Ry = grid_height / (2.0*K_SI*T_CHIP*grid_width);
	double Rz = T_CHIP / (K_SI*grid_height*grid_width);
	double max_slope = MAX_PD / (FACTOR_CHIP*T_CHIP*SPEC_HEAT_SI);
	double step = PRECISION / max_slope;
	
	// Initialising memory
	printf("\n[hotspot] Initialising memory"); fflush(stdout);
	double temperature[GRID_ROWS][GRID_COLS];
	double power[GRID_ROWS][GRID_COLS];
	double result[GRID_ROWS][GRID_COLS];
	
	// Read initial temperature array
	printf("\n[hotspot] Populating memory"); fflush(stdout);
	char string[STRING_SIZE];
	double value;
	FILE* file_pointer1 = fopen(TEMPERATURE_FILE, "r");
	if (!file_pointer1) { printf("\n[hotspot] Error: file '%s' could not be opened for reading\n\n", TEMPERATURE_FILE); fflush(stdout); exit(1); }
	for (r=0; r<GRID_ROWS; r++) {
		for (c=0; c<GRID_COLS; c++) {
			fgets(string, STRING_SIZE, file_pointer1);
			if (feof(file_pointer1))                    { printf("\n[hotspot] Error: not enough lines in file '%s'\n\n", TEMPERATURE_FILE); fflush(stdout); exit(1); }
			if ((sscanf(string, "%lf", &value) != 1) ) { printf("\n[hotspot] Error: invalid file format for '%s'\n\n", TEMPERATURE_FILE); fflush(stdout); exit(1); }
			temperature[r][c] = value;
		}
	}
	fclose(file_pointer1);

	// Read initial power array
	FILE* file_pointer2 = fopen(POWER_FILE, "r");
	if (!file_pointer2) { printf("\n[hotspot] Error: file '%s' could not be opened for reading\n\n", POWER_FILE); fflush(stdout); exit(1); }
	for (r=0; r<GRID_ROWS; r++) {
		for (c=0; c<GRID_COLS; c++) {
			fgets(string, STRING_SIZE, file_pointer2);
			if (feof(file_pointer2))                    { printf("\n[hotspot] Error: not enough lines in file '%s'\n\n", POWER_FILE); fflush(stdout); exit(1); }
			if ((sscanf(string, "%lf", &value) != 1) ) { printf("\n[hotspot] Error: invalid file format for '%s'\n\n", POWER_FILE); fflush(stdout); exit(1); }
			power[r][c] = value;
		}
	}
	fclose(file_pointer2);
	
	// Perform the computation a given number of times
	printf("\n[hotspot] Performing the computation %d times",SIM_TIME); fflush(stdout);
	#pragma scop
	for (iter=0; iter<SIM_TIME; iter++) {
		
		// Transient solver driver routine: convert the heat transfer differential equations to difference equations
		// and solve the difference equations by iterating
		for (r=0; r<GRID_ROWS; r++) {
			for (c=0; c<GRID_COLS; c++) {

				// Load the temperatures from memory
				double temp_r_c = temperature[r][c];
				double temp_rp1_c = 0;
				if (r < GRID_ROWS-1) { temp_rp1_c = temperature[r+1][c]; }
				double temp_rm1_c = 0;
				if (r > 0) { temp_rm1_c = temperature[r-1][c]; }
				double temp_r_cp1 = 0;
				if (c < GRID_COLS-1) { temp_r_cp1 = temperature[r][c+1]; }
				double temp_r_cm1 = 0;
				if (c > 0) { temp_r_cm1 = temperature[r][c-1]; }

				// Load the power
				double power_r_c = power[r][c];
				
				// Corner 1
				if ( (r == 0) && (c == 0) ) {
					delta = (step / cap) * (power_r_c + 
						(temp_r_cp1 - temp_r_c) / Rx + 
						(temp_rp1_c - temp_r_c) / Ry + 
						(AMB_TEMP   - temp_r_c) / Rz);
				}
				// Corner 2
				else if ((r == 0) && (c == GRID_COLS-1)) {
					delta = (step / cap) * (power_r_c + 
						(temp_r_cm1 - temp_r_c) / Rx + 
						(temp_rp1_c - temp_r_c) / Ry + 
						(AMB_TEMP   - temp_r_c) / Rz);
				}
				// Corner 3
				else if ((r == GRID_ROWS-1) && (c == GRID_COLS-1)) {
					delta = (step / cap) * (power_r_c + 
						(temp_r_cm1 - temp_r_c) / Rx + 
						(temp_rm1_c - temp_r_c) / Ry + 
						(AMB_TEMP   - temp_r_c) / Rz);
				}
				// Corner 4
				else if ((r == GRID_ROWS-1) && (c == 0)) {
					delta = (step / cap) * (power_r_c + 
						(temp_r_cp1 - temp_r_c) / Rx + 
						(temp_rm1_c - temp_r_c) / Ry + 
						(AMB_TEMP   - temp_r_c) / Rz);
				}
				// Edge 1
				else if (r == 0) {
					delta = (step / cap) * (power_r_c + 
						(temp_r_cp1 + temp_r_cm1 - 2.0*temp_r_c) / Rx + 
						(temp_rp1_c              -     temp_r_c) / Ry + 
						(AMB_TEMP                -     temp_r_c) / Rz);
				}
				// Edge 2
				else if (c == GRID_COLS-1) {
					delta = (step / cap) * (power_r_c + 
						(temp_rp1_c + temp_rm1_c - 2.0*temp_r_c) / Ry + 
						(temp_r_cm1              -     temp_r_c) / Rx + 
						(AMB_TEMP                -     temp_r_c) / Rz);
				}
				// Edge 3
				else if (r == GRID_ROWS-1) {
					delta = (step / cap) * (power_r_c + 
						(temp_r_cp1 + temp_r_cm1 - 2.0*temp_r_c) / Rx + 
						(temp_rm1_c              -     temp_r_c) / Ry + 
						(AMB_TEMP                -     temp_r_c) / Rz);
				}
				// Edge 4
				else if (c == 0) {
					delta = (step / cap) * (power_r_c + 
						(temp_rp1_c + temp_rm1_c - 2.0*temp_r_c) / Ry + 
						(temp_r_cp1              -     temp_r_c) / Rx + 
						(AMB_TEMP                -     temp_r_c) / Rz);
				}
				// Inside the chip
				else {
					delta = (step / cap) * (power_r_c + 
						(temp_rp1_c + temp_rm1_c - 2.0*temp_r_c) / Ry + 
						(temp_r_cp1 + temp_r_cm1 - 2.0*temp_r_c) / Rx + 
						(AMB_TEMP                -     temp_r_c) / Rz);
				}
			
			// Update the temperatures
			result[r][c] = temperature[r][c] + delta;
			}
		}
		
		// Copy the result as the new temperatures
		for (r=0; r<GRID_ROWS; r++) {
			for (c=0; c<GRID_COLS; c++) {
				temperature[r][c] = result[r][c];
			}
		}
	}
	#pragma endscop
	
	// Print the values matrix
	//printf("\n[hotspot] Printing the final temperatures:\n\n"); fflush(stdout);
	//for (r=0; r<GRID_ROWS; r++) {
	//	for (c=0; c<GRID_COLS; c++) {
	//		printf("%6d: %.3lf ", r*GRID_COLS+c, temperature[r][c]);
	//	}
	//	printf("\n");
	//}
	
	// Clean-up and exit
	printf("\n[hotspot] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}

//########################################################################
//### Function to read an input file (power or temperature values)
//########################################################################

void read_input(double array[GRID_ROWS][GRID_COLS], const char* filename) {

}

//########################################################################