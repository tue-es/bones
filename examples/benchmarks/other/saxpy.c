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
// Last modified on...09-Aug-2012
//

#include <stdio.h>
#include <stdlib.h>
#define N 2048*2048

// This is 'saxpy', a scalar multiplication and vector addition kernel
int main(void) {
	int i;
	
	// Declare arrays on the stack
	float x[N];
	float y[N];
	
	// Set the input data
	for (i=0; i<N; i++) {
		x[i] = i*1.4;
		y[i] = i/0.9;
	}
	
	// Set the constants
	float a = 411.3;
	
	// Perform the computation (y := ax+y)
	#pragma scop
	#pragma species kernel 0:N-1|element ^ 0:N-1|element -> 0:N-1|element
	for (i=0; i<N; i++) {
		y[i] = a*x[i] + y[i];
	}
	#pragma species endkernel saxpy
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

