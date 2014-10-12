//
// This file is part of the Bones source-to-source compiler examples. This C-code
// example is meant to illustrate the use of Bones. For more information on Bones
// use the contact information below.
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........fusion/example04.c
// Author.............Cedric Nugteren
// Last modified on...11-October-2014
//

#include <stdio.h>

// This is 'example04', with code similar to PolyBench's "atax" benchmark
int main(void) {
	int i,j;
	
	// Declare arrays on the stack
	float A[4096][4096];
	float x[4096];
	float y[4096];
	float tmp[4096];
	
	// Set the input data
	for (i=0; i<4096; i++) {
		x[i] = i*3.14159;
	}
	for (i=0; i<4096; i++) {
		for (j=0; j<4096; j++) {
			A[i][j] = ((float) i*(j+1)) / 4096;
		}
	}
	
	// Perform the computation (y := A'Ax)
	#pragma scop
	#pragma species kernel A[0:4095,0:4095]|chunk(0:0,0:4095) ^ x[0:4095]|full -> tmp[0:4095]|element
	for (i=0; i<4096; i++) {
		tmp[i] = 0;
		for (j=0; j<4096; j++) {
			tmp[i] = tmp[i] + A[i][j] * x[j];
		}
	}
	#pragma species endkernel atax-part1
	#pragma species kernel A[0:4095,0:4095]|chunk(0:4095,0:0) ^ tmp[0:4095]|full -> y[0:4095]|element
	for (j=0; j<4096; j++) {
		y[j] = 0;
		for (i=0; i<4096; i++) {
			y[j] = y[j] + A[i][j] * tmp[i];
		}
	}
	#pragma species endkernel atax-part2
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	y[9] = y[9];
	return 0;
}

