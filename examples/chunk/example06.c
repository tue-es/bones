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
// Filename...........chunk/example06.c
// Author.............Cedric Nugteren
// Last modified on...07-May-2013
//

#include <stdio.h>

// This is 'example06', a chunk/chunk with a step of 2
int main(void) {
	int i;
	int N = 256;
	
	// Declare input/output arrays
	int A[N];
	int B[N];
	
	// Set the input data
	for(i=0;i<N;i++) {
		A[i] = i;
		B[i] = i+5;
	}
	
	// Perform the computation
	#pragma species kernel A[2:N-1]|chunk(0:1) -> B[2:N-1]|chunk(0:1)
	for (i=2; i<N-1; i=i+2) {
		B[i] = A[i];
		B[i+1] = A[i+1];
	}
	#pragma species endkernel example06
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

