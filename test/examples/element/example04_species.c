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
// Filename...........element/example4.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>

// This is 'example4', demonstrating two input arrays and an inner-loop as the computational body
int main(void) {
	int i,l;
	float factor = 0;
	
	// Declare input/output arrays
	float A[700];
	float B[700];
	float C[700];
	
	// Set the input data
	for(i=0;i<700;i++) {
		A[i] = i*2.3;
		B[i] = i+6.0;
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:699]|element ^ B[0:699]|element -> C[0:699]|element
		for (i = 0; i < 700; i++) {
			factor = 0.5;
			for (l = 0; l < 3; l++) {
				factor = factor + 0.2 * factor;
			}
			C[i] = factor * A[i] + factor * B[i];
		}
		#pragma species endkernel example04_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

