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
// Filename...........shared/example4.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#include <stdlib.h>
#define SIZE 1024*1024

// This is 'example4', demonstrating a basic 256-bin histogram computation
int main(void) {
	int i;
	unsigned char index;
	
	// Declare input/output arrays
	unsigned char *A = (unsigned char *)malloc(SIZE*sizeof(unsigned char));
	int B[256];
	
	// Set the input data
	for(i=0;i<SIZE;i++) {
		A[i] = i%256;
	}
	
	// Set the output to zero before starting
	for (i=0;i<256;i++) {
		B[i] = 0;
	}
	
	// Perform the computation
	#pragma scop
	{
		#pragma species kernel A[0:SIZE-1]|element ^ B[0:255]|full -> B[0:255]|shared
		for (i = 0; i < SIZE; i++) {
			index = A[i];
			B[index]++;
		}
		#pragma species endkernel example04_k1
	}
	#pragma endscop
	
	// Clean-up and exit the function
	free(A);
	fflush(stdout);
	return 0;
}

