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
// Filename...........neighbourhood/example4.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>

// This is 'example4', demonstrating naming (optional) in the classification to distingish the two input arrays
int main(void) {
	int i;
	float factor;
	int size = 512;
	
	// Declare input/output arrays
	float A[size];
	float B[size];
	float C[size];
	
	// Set the input data
	for(i=0;i<size;i++) {
		A[i] = i*2.3;
		B[i] = i+6.0;
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:size-1]|element ^ B[0:size-1]|neighbourhood(-1:1) -> C[0:size-1]|element
		for (i = 0; i < size; i++) {
			factor = A[i] / 100.0;
			if (i >= 1 && i < size - 1) {
				C[i] = factor * (B[i - 1] + B[i] + B[i + 1]);
			} else {
				C[i] = B[i];
			}
		}
		#pragma species endkernel example04_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

