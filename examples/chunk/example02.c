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
// Filename...........chunk/example02.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>
#define SIZE 2048
#define HALFSIZE (SIZE/2)

// This is 'example02', demonstrating a chunk-example without an inner-loop, everything is unrolled manually
int main(void) {
	int i;
	
	// Declare input/output arrays
	float A[SIZE];
	float B[HALFSIZE];
	
	// Set the input data
	for(i=0;i<SIZE;i++) {
		A[i] = i%6+i;
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel A[0:SIZE-1]|chunk(0:1) -> B[0:HALFSIZE-1]|element
	for(i=0;i<HALFSIZE;i++) {
		B[i] = A[i*2] + A[i*2+1];
	}
	#pragma species endkernel example2
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

