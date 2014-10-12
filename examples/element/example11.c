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
// Filename...........element/example11.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>

// This is 'example11', demonstrating an inner-loop which is dependent on an outer-loop variable and a classification of the inner-loop only
int main(void) {
	int i,j;
	
	// Declare input/output arrays
	int A[128][128];
	int B[128][128];
	
	// Set the input data
	for(i=0;i<128;i++) {
		for(j=0;j<128;j++) {
			A[i][j] = i+j;
			B[i][j] = 999;
		}
	}
	
	// Perform the computation
	#pragma scop
	for(i=0;i<128;i++) {
		#pragma species kernel A[i:i,i:127]|element -> B[i:i,i:127]|element
		for(j=i;j<128;j++) {
			B[i][j] = 2*A[i][j];
		}
		#pragma species endkernel example11
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

