//
// This file is part of the Bones source-to-source compiler examples. For more
// information on Bones please use the contact information below.
//
// == More information on Bones
// Contact............Cedric Nugteren <c.nugteren@tue.nl>
// Web address........http://parse.ele.tue.nl/bones/
//
// == File information
// Filename...........benchmark/saxpy.c
// Author.............Cedric Nugteren
// Last modified on...04-Jul-2012
//

#include "common.h"

// This is 'saxpy', a scalar multiplication and vector addition kernel
int main(void) {
	int i;
	
	// Declare arrays on the stack
	float x[LARGE_N];
	float y[LARGE_N];
	
	// Set the input data
	for (i=0; i<LARGE_N; i++) {
		x[i] = i*1.4;
		y[i] = i/0.9;
	}
	
	// Set the constants
	float a = 411.3;
	
	// Perform the computation (y := ax+y)
	#pragma species kernel 0:LARGE_N-1|element ^ 0:LARGE_N-1|element -> 0:LARGE_N-1|element
	for (i=0; i<LARGE_N; i++) {
		y[i] = a*x[i] + y[i];
	}
	#pragma species endkernel saxpy
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

