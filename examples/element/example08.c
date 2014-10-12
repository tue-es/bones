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
// Filename...........element/example08.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>

// This is 'example08', demonstrating a reading and writing from the same array
int main(void) {
	int i,j;
	
	// Declare input/output arrays
	int A[100][16];
	
	// Set the input data
	for(i=0;i<100;i++) {
		for(j=0;j<16;j++) {
			A[i][j] = i+j;
		}
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel A[0:99,0:15]|element -> A[0:99,0:15]|element
	for(i=0;i<100;i++) {
		for(j=0;j<16;j++) {
			A[i][j] = 2*A[i][j];
		}
	}
	#pragma species endkernel example8
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

