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
// Filename...........element/example12.c
// Author.............Cedric Nugteren
// Last modified on...06-Aug-2012
//

#include <stdio.h>
#include <stdlib.h>

void computation(int* A, int* B, int constant);

// This is 'example12', demonstrating a classification in another function
int main(void) {
	int i;
	
	// Declare input/output arrays
	int* A = (int*)malloc(128*sizeof(int));
	int* B = (int*)malloc(128*sizeof(int));
	
	// Set the input data
	for(i=0;i<128;i++) {
		A[i] = i+3;
		B[i] = 999;
	}
	int constant = 3;
	
	// Call the computation function
	computation(A,B,constant);
	
	// Clean-up and exit the function
	free(A);
	free(B);
	fflush(stdout);
	return 0;
}

// Function implementing the computation for 'example12'
void computation(int* A, int* B, int constant) {
	int i;
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:127]|element -> B[0:127]|element
		for (i = 0; i < 128; i++) {
			B[i] = 2 * A[i] + constant;
		}
		#pragma species endkernel example12_k1
	}
	#pragma endscop
}
