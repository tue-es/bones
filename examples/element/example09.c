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
// Filename...........element/example09.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>
#include <stdlib.h>

// This is 'example09', demonstrating a for-loop that does not start at zero
int main(void) {
	int i;
	int result;
	int N = 2048*2048;
	
	// Declare input/output arrays
	int *A = (int *)malloc(N*sizeof(int));
	int *B = (int *)malloc(N*sizeof(int));
	
	// Set the input data
	for(i=0;i<N;i++) {
		A[i] = i;
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel A[2:N-1]|element -> B[2:N-1]|element
	for(i=2;i<N;i++) {
		result = A[i] + 3;
		B[i] = result * 3;
	}
	#pragma species endkernel example9
	#pragma endscop
	
	// Clean-up and exit the function
	free(A);
	free(B);
	fflush(stdout);
	return 0;
}

