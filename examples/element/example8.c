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
// Filename...........element/example8.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>

// This is 'example8', demonstrating a reading and writing from the same array
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
	#pragma species kernel 0:99,0:15|element -> 0:99,0:15|element
	for(i=0;i<100;i++) {
		for(j=0;j<16;j++) {
			A[i][j] = 2*A[i][j];
		}
	}
	#pragma species endkernel example8
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}
