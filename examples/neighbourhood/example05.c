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
// Filename...........neighbourhood/example05.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>

// This is 'example05', an unrolled one-sided neighbourhood
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
	#pragma scop
	#pragma species kernel A[2:N]|neighbourhood(0:1) -> B[2:N-1]|element
	for (i=2; i<N; i++) {
		B[i] = A[i] + A[i+1];
	}
	#pragma species endkernel example05
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

