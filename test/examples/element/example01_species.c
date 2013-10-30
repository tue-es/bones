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
// Filename...........element/example1.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>

// This is 'example1', a very basic element to element example using 2D arrays.
int main(void) {
	int i,j;
	
	// Declare input/output arrays
	int A[100][16];
	int B[100][16];
	
	// Set the input data
	for(i=0;i<100;i++) {
		for(j=0;j<16;j++) {
			A[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:99,0:15]|element -> B[0:99,0:15]|element
		for (i = 0; i < 100; i++) {
			for (j = 0; j < 16; j++) {
				B[i][j] = 2 * A[i][j];
			}
		}
		#pragma species endkernel example01_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

