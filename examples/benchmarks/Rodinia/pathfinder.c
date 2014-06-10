//
// This file is part of the Bones source-to-source compiler examples. This C-code
// demonstrates the use of Bones for an example application: 'pathfinder', taken from
// the Rodinia benchmark suite. For more information on the application or on Bones
// please use the contact information below.
//
// == More information on Hotspot
// Original code......https://www.cs.virginia.edu/~skadron/wiki/rodinia/
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........applications/pathfinder.c
// Authors............Cedric Nugteren
// Last modified on...05-Jun-2014
//
//########################################################################

// Includes
#include "common.h"

//########################################################################
//### Start of the main function
//########################################################################

int main(void) {

	// Variables
	unsigned long long cycles;
	int min;

	// Initialize arrays
	int wall[PATHROWS][PATHCOLS];
	int result[PATHCOLS];
	int input[PATHCOLS];

	// Seed
	int seed = M_SEED;
	srand(seed);

	// Set initial values
	for (int i=0; i<PATHROWS; i++) {
		for (int j=0; j<PATHCOLS; j++) {
			wall[i][j] = rand() % 10;
		}
	}
	for (int j=0; j<PATHCOLS; j++) {
		result[j] = wall[0][j];
	}
	
	// Iterate over the PATHROWS
	#pragma scop
	for (int t=1; t<PATHROWS; t++) {

		// Copy result of previous iteration as current input
		for (int n=0; n<PATHCOLS; n++) {
			input[n] = result[n];
		}

		// Iterate over the columns
		for (int n=0; n<PATHCOLS; n++) {
			min = input[n];
			if (n > 0) {
				int val1 = input[n-1];
				if (val1 < min) {
					min = val1;
				}
			}
			if (n < PATHCOLS-1) {
				int val2 = input[n+1];
				if (val2 < min) {
					min = val2;
				}
			}
			result[n] = wall[t][n] + min;
		}
	}
	#pragma endscop
	
	// Clean-up and exit
	printf("\n[pathfinder] Completed\n\n"); fflush(stdout);
	fflush(stdout);
	return 0;
}

//########################################################################