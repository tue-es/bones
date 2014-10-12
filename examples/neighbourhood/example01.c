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
// Filename...........neighbourhood/example01.c
// Author.............Cedric Nugteren
// Last modified on...10-October-2014
//

#include <stdio.h>
#define SIZE 60000
#define NB 2

// This is 'example01', demonstrating a basic 1D neighbourhood-based computation whose size is set by a define
int main(void) {
	int i,n;
	float result = 0;
	
	// Declare input/output arrays
	float A[SIZE];
	float B[SIZE];
	
	// Set the input data
	for(i=0;i<SIZE;i++) {
		A[i] = i/2.0;
	}
	
	// Perform the computation
	#pragma scop
	#pragma species kernel A[0:SIZE-1]|neighbourhood(-NB:NB) -> B[0:SIZE-1]|element
	for(i=0;i<SIZE;i++) {
		if (i >= NB && i < SIZE-NB) {
			result = 0;
			for (n=-NB;n<=NB;n++) {
				result = result + A[i+n];
			}
			B[i] = result / (NB*2+1);
		}
		else {
			B[i] = A[i];
		}
	}
	#pragma species endkernel example1
	#pragma endscop
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

