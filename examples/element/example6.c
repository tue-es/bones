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
// Filename...........element/example6.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#include <stdlib.h>

// This is 'example6', demonstrating dynamically sized arrays, a dynamically sized kernel, and a classification including variables
int main(void) {
	int i;
	int N = 2048*2048;
	
	// Declare input/output arrays
	int *A = (int *)malloc(N*sizeof(int));
	int *B = (int *)malloc(N*sizeof(int));
	
	// Set the input data
	for(i=0;i<N;i++) {
		A[i] = i;
	}
	
	// Perform the computation
	#pragma species kernel 0:N-1|element -> 0:N-1|element
	for(i=0;i<N;i++) {
		B[i] = A[i] + 3;
	}
	#pragma species endkernel example6
	
	// Clean-up and exit the function
	free(A);
	free(B);
	fflush(stdout);
	return 0;
}

