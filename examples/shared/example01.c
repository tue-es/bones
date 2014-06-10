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
// Filename...........shared/example1.c
// Author.............Cedric Nugteren
// Last modified on...16-April-2012
//

#include <stdio.h>
#include <stdlib.h>
#define SIZE 512*1024

// This is 'example1', a basic associative and commutative reduction to scalar
int main(void) {
	int i;
	
	// Declare input/output arrays
	int *A = (int *)malloc(SIZE*sizeof(int));
	int B[1];
	
	// Set the input data
	for(i=0;i<SIZE;i++) {
		A[i] = 1;
	}
	
	// Perform the computation
	B[0] = 0;
	#pragma species kernel 0:SIZE-1|element -> 0:0|shared
	for(i=0;i<SIZE;i++) {
		B[0] = B[0] + A[i];
	}
	#pragma species endkernel example1
	
	// Clean-up and exit the function
	fflush(stdout);
	return 0;
}

