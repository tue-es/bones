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
// Filename...........dependences/example02.c
// Author.............Cedric Nugteren
// Last modified on...11-October-2014
//

#include <stdio.h>

// This is 'example02', an element -> element example WITH dependences.
int main(void) {
	int i;
	int N = 256;
	
	// Declare input/output arrays
	int A[N];
	
	// Set the input data
	for(i=0;i<N;i++) {
		A[i] = i;
	}
	
	// Perform the computation
	#pragma scop
	for (i=0; i<N; i++) { // Read-write dependence
		A[2*i+6] = A[3*i+1];
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

